from __future__ import print_function, absolute_import
import torchvision.utils as vutils
from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss, TupletLoss
from .utils.meters import AverageMeter
from lz import *
from tensorboardX import SummaryWriter
from reid.mining import mine_hard_triplets
import torch
from .models.resnet import get_loss_div


class BaseTrainer(object):
    def __init__(self, model, criterion, dbg=False, logs_at='work/vis'):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.dbg = dbg
        self.iter = 0

        if dbg:
            mkdir_p(logs_at, delete=True)
            self.writer = SummaryWriter(logs_at)

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)
            if isinstance(targets, tuple):
                targets, _ = targets
            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
        return collections.OrderedDict({
            'ttl-time': batch_time.avg,
            'data-time': data_time.avg,
            'loss': losses.avg,
            'prec': precisions.avg
        })

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class VerfTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, fnames, pids, = inputs.get('img'), inputs.get('fname'), inputs.get('pid')
        inputs = [Variable(imgs.cuda(), requires_grad=False)]
        targets = Variable(pids.cuda(), requires_grad=False)
        return inputs, (targets, None)

    def _forward(self, inputs, targets):
        targets, info = targets
        # self.model.eval()
        pred, y, info = self.model(inputs[0], targets, info)

        loss = self.criterion(pred, y)
        if len(pred.shape)==2:
            pred = pred.data[:,0]
        else:
            pred = pred.data
        right = (to_numpy(pred  > self.criterion.margin / 2.).reshape(-1) == to_numpy(y.data).reshape(-1))
        prec1 = (right.astype(float).sum() / right.shape[0])
        # print(prec1)
        return loss, prec1


class TripletTrainer(object):
    def __init__(self, model, criterion, dbg=False, logs_at='work/vis'):
        self.model = model
        self.criterion = criterion
        self.dbg = dbg
        self.iter = 0

        if dbg:
            mkdir_p(logs_at, delete=True)
            self.writer = SummaryWriter(logs_at)

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)
            if isinstance(targets, tuple):
                targets, _ = targets
            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
        return collections.OrderedDict({
            'ttl-time': batch_time.avg,
            'data-time': data_time.avg,
            'loss': losses.avg,
            'prec': precisions.avg
        })

    def _parse_data(self, inputs):
        def _parse_one(inputs):
            imgs, npys, fnames, pids = inputs.get('img'), inputs.get('npy'), inputs.get('fname'), inputs.get('pid')
            # print(fnames)
            return imgs

        # cvb.dump(np.asarray([in_.get('fname') for in_ in inputs ]).ravel(), 'tmp.pkl')
        a, p, n = _parse_one(inputs[0]), _parse_one(inputs[1]), _parse_one(inputs[2])
        inputs = to_variable([a, p, n])
        targets = to_variable(torch.ones(len(a)))
        return inputs, targets

    def _forward(self, inputs, targets):
        dist_an, dist_ap = self.model(*inputs)
        loss = self.criterion(dist_an, dist_ap, targets)
        # ByteTensor has no mean() method
        prec = (dist_an.data > dist_ap.data).sum() * 1. / targets.size(0)
        return loss, prec


class SiameseTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        (imgs1, _, pids1, _), (imgs2, _, pids2, _) = inputs
        imgs1, imgs2 = inputs[0].get('img'), inputs[1].get('img')
        pids1, pids2 = inputs[0].get('pid'), inputs[1].get('pid')
        inputs = [Variable(imgs1), Variable(imgs2)]
        targets = Variable((pids1 == pids2).long().cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        loss = self.criterion(outputs, targets)
        prec1, = accuracy(outputs.data, targets.data)
        return loss, prec1[0]


def stat_(writer, tag, tensor, iter):
    writer.add_scalars('groups/' + tag, {
        'mean': torch.mean(tensor),
        'media': torch.median(tensor),
        'min': torch.min(tensor),
        'max': torch.max(tensor)
    }, iter)


class Trainer(object):
    def __init__(self, model, criterion, dbg=False, logs_at='work/vis', loss_div_weight=0):
        self.loss_div_weight = loss_div_weight
        self.model = model
        self.criterion = criterion
        self.dbg = dbg
        self.iter = 0
        if dbg:
            mkdir_p(logs_at, delete=True)
            self.writer = SummaryWriter(logs_at)

    def _parse_data(self, inputs):
        imgs, npys, fnames, pids = inputs.get('img'), inputs.get('npy'), inputs.get('fname'), inputs.get('pid')
        inputs = [imgs, npys]
        inputs = to_variable(inputs, requires_grad=False)
        targets = to_variable(pids, requires_grad=False)
        return inputs, targets, fnames

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        if self.model.module.dconv_model is not None and hasattr(self.model.module.dconv_model, 'weight'):
            weight = self.model.module.dconv_model.weight
            weight = weight.view(weight.size(0), -1)
            loss_div = get_loss_div(weight)
        else:
            loss_div = 0.
        if self.dbg and self.iter % 1000 == 0:
            self.writer.add_histogram('1_input', inputs[0], self.iter)
            self.writer.add_histogram('2_feature', outputs, self.iter)
            x = vutils.make_grid(to_torch(inputs[0]), normalize=True, scale_each=True)
            self.writer.add_image('input', x, self.iter)

        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            loss += loss_div * self.loss_div_weight
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            if self.dbg and self.iter % 10 == 0:
                loss, prec, dist, dist_ap, dist_an = self.criterion(outputs, targets, dbg=self.dbg)
                diff = dist_an - dist_ap
                self.writer.add_histogram('an-ap', diff, self.iter)

                # stat_(self.writer, 'an-ap', diff, self.iter)
                self.writer.add_scalar('vis/loss', loss - loss_div * self.loss_div_weight, self.iter)
                self.writer.add_scalar('vis/loss_div', loss_div, self.iter)
                self.writer.add_scalar('vis/loss_ttl', loss, self.iter)
                self.writer.add_scalar('vis/prec', prec, self.iter)
                self.writer.add_histogram('dist', dist, self.iter)
                self.writer.add_histogram('ap', dist_ap, self.iter)
                self.writer.add_histogram('an', dist_an, self.iter)
            else:
                loss, prec = self.criterion(outputs, targets, dbg=False)

        elif isinstance(self.criterion, TupletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        self.iter += 1
        return loss, prec

    def train(self, epoch, data_loader, optimizer, print_freq=5):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        # triplets = mine_hard_triplets(self.model, data_loader, margin=0.5)
        # al_ind = np.asarray(triplets).flatten()
        # bins, _ = np.histogram(al_ind, bins=data_loader.sampler.info.shape[0],
        #                        range=(0, data_loader.sampler.info.shape[0]))
        # bins += 1
        # bins = bins / bins.sum()
        # data_loader.sampler.update_weight(bins)

        # if np.random.rand(1) < 0.005:
        #     print('global probs ', np.asarray(data_loader.sampler.info['probs']))
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            inputs, targets, fnames = self._parse_data(inputs)

            # global_inds = [data_loader.fname2ind[fn] for fn in fnames]

            # triplets = mine_hard_triplets(self.model, data_loader, margin=0.5)
            # al_ind = np.asarray(triplets).flatten()
            # bins, divs = np.histogram(al_ind, bins=data_loader.sampler.info.shape[0],
            #                        range=(0, data_loader.sampler.info.shape[0]))
            # bins += 1
            # bins = bins / bins.sum()
            # data_loader.sampler.update_weight(bins)
            # print('global probs ', np.asarray(data_loader.sampler.info['probs']))

            self.model.train()
            loss, prec1 = self._forward(inputs, targets)
            if isinstance(targets, tuple):
                targets, _ = targets
            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not torch.cuda.is_available():
                print('one f!')
                del inputs, targets, loss, prec1
                import gc
                gc.collect()
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
        return collections.OrderedDict({
            'ttl-time': batch_time.avg,
            'data-time': data_time.avg,
            'loss': losses.avg,
            'prec': precisions.avg
        })
