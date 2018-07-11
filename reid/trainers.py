import torchvision.utils as vutils
from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss, TupletLoss
from .utils.meters import AverageMeter
from lz import *
from tensorboardX import SummaryWriter
from reid.mining import mine_hard_triplets
import torch
from .models.resnet import *


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
        else:
            self.writer = None

    def _parse_data(self, inputs):
        imgs, npys, fnames, pids = inputs.get('img'), inputs.get(
            'npy'), inputs.get('fname'), inputs.get('pid')
        inputs = [imgs, npys]
        inputs = to_variable(inputs, requires_grad=False)
        targets = to_variable(pids, requires_grad=False)
        return inputs, targets, fnames

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        if self.model.module.dconv_model is not None and hasattr(self.model.module.dconv_model, 'weight'):
            weight = self.model.module.dconv_model.weight
            weight = weight.view(weight.size(0), -1)
            # loss_div = get_loss_div(weight)
            loss_div = 0.
        else:
            loss_div = 0.
        if self.dbg and self.iter % 1000 == 0:
            self.writer.add_histogram('1_input', inputs[0], self.iter)
            self.writer.add_histogram('2_feature', outputs, self.iter)
            x = vutils.make_grid(
                to_torch(inputs[0]), normalize=True, scale_each=True)
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
            if self.dbg and self.iter % 100 == 0:
                loss, prec, dist, dist_ap, dist_an = self.criterion(
                    outputs, targets, dbg=self.dbg)
                diff = dist_an - dist_ap
                self.writer.add_histogram('an-ap', diff, self.iter)

                # stat_(self.writer, 'an-ap', diff, self.iter)
                self.writer.add_scalar(
                    'vis/loss', loss - loss_div * self.loss_div_weight, self.iter)
                self.writer.add_scalar('vis/loss_div', loss_div, self.iter)
                self.writer.add_scalar('vis/loss_ttl', loss, self.iter)
                self.writer.add_scalar('vis/prec', prec, self.iter)
                self.writer.add_histogram('dist', dist, self.iter)
                self.writer.add_histogram('ap', dist_ap, self.iter)
                self.writer.add_histogram('an', dist_an, self.iter)
                self.writer.add_scalar('vis/lr', self.lr,
                                       self.iter)  # schedule.get_lr()
            else:
                loss, prec = self.criterion(outputs, targets, dbg=False)
        elif isinstance(self.criterion, TupletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        self.iter += 1
        return loss, prec

    def train(self, epoch, data_loader, optimizer, print_freq=5, schedule=None):

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
            if schedule is not None:
                schedule.batch_step()
            # print('lr is ', optimizer.param_groups[0]['lr'])
            self.lr = optimizer.param_groups[0]['lr']
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
            # self.model.zero_grad()
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


def update_dop_cls(outputs, targets, dop_file):
    targets = to_numpy(targets)
    targets = targets.reshape(
        (targets.shape[0] // 4), 4).mean(axis=1).astype(np.int64)

    outputs = to_numpy(outputs)
    outputs = outputs.reshape(
        (outputs.shape[0] // 4, 4, outputs.shape[1])).sum(axis=1)

    outputs[np.arange(outputs.shape[0]), targets] = -np.inf

    db = Database(dop_file, 'w')
    dop = db['dop']
    dop[targets] = np.argmax(outputs, axis=1)
    logging.debug('cls \n {} dop is \n {}'.format(targets, dop[targets]))
    db['dop'] = dop
    db.close()


def update_dop_tri2(dist, targets, dop_info):
    bs = targets.shape[0] // 4
    # todo select max
    raise NotImplementedError('wait ,,,')


def update_dop_tri(dist, targets, dop_info):
    # todo on gpu operation, compare speed
    # todo mean not depent on 4
    targets = to_numpy(targets)
    bs = targets.shape[0] // 4
    targets = targets.reshape(bs, 4).mean(axis=1).astype(np.int64)
    dist = to_numpy(dist)
    dist = dist.reshape(bs, 4, bs, 4)
    dist = np.transpose(dist, (0, 2, 1, 3)).reshape(bs, bs, 16).sum(axis=2)
    dist += np.diag([np.inf] * bs)

    dop = dop_info.dop
    dop[targets] = targets[np.argmin(dist, axis=1)]
    dop_info.dop = dop


def update_dop_center(dist, dop_info):
    num_classes = dist.shape[0]
    dist_new = dist + \
               torch.diag(torch.ones(num_classes)).cuda() * torch.max(dist)
    dop = dop_info.dop
    dop[:] = to_numpy(torch.argmin(dist_new, dim=0))
    dop_info.dop = dop


def set_bn_to_eval(m):
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


class TriTrainer(object):
    def __init__(self, model, criterion, logs_at='work/vis', dbg=True, args=None, dop_info=None, **kwargs):
        self.model = model
        self.criterion = criterion[0]
        self.args = args
        self.dop_info = dop_info  # deprecated
        self.iter = 0
        self.dbg = dbg
        if dbg:
            mkdir_p(logs_at, delete=True)
            self.writer = SummaryWriter(logs_at)
        else:
            self.writer = None

    def train(self, epoch, data_loader, optimizer, print_freq=5, schedule=None):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        if self.args.freeze_bn:
            self.model.train()
            self.model.apply(set_bn_to_eval)
        else:
            self.model.train()

        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            input_imgs = inputs.get('img').cuda()
            batch_size = input_imgs.size(0)
            if self.dbg % 100 == 0:
                x = vutils.make_grid(
                    input_imgs, normalize=True, scale_each=True)
                self.writer.add_image('input', x, self.iter)

            input_imgs.requires_grad_(True)
            targets = inputs.get('pid').cuda()
            if schedule is not None:
                schedule.batch_step()
            self.lr = optimizer.param_groups[0]['lr']

            features, logits, mid_feas, x5_grad_reg = self.model(input_imgs)
            x5_grad_reg.squeeze_(dim=0)
            mid_feas.append(features)
            losst, prect, dist_tri = self.criterion(features, targets, dbg=False)
            # losst is triplet loss

            self.writer.add_scalar('vis/loss-triplet', losst.item(), self.iter)
            self.writer.add_scalar('vis/prec-triplet', prect.item(), self.iter)
            self.writer.add_scalar('vis/lr', self.lr, self.iter)
            self.iter += 1

            if isinstance(targets, tuple):
                targets, _ = targets
            losses.update(losst.item(), targets.size(0))
            precisions.update(prect.item(), targets.size(0))

            if math.isnan(losst.item()):
                raise ValueError(f'nan {losst}')
            if self.args.adv_inp != 0 and self.args.double == 0:
                # method1
                # optimizer.zero_grad()
                # if self.iter % 2 == 0:
                #     losst.backward()
                # else:
                #     input_imgs_grad_detached = torch.autograd.grad(
                #         outputs=losst, inputs=input_imgs,
                #         create_graph=True, retain_graph=True,
                #         only_inputs=True
                #     )[0].detach()
                #     if self.args.aux == 'l2_adv':
                #         input_imgs_adv = input_imgs + self.args.adv_inp_eps * l2_normalize(input_imgs_grad_detached)
                #     elif self.args.aux == 'linf_adv':
                #         input_imgs_adv = input_imgs + self.args.adv_inp_eps * torch.sign(input_imgs_grad_detached)
                #     else:
                #         input_imgs_adv = input_imgs + self.args.adv_inp_eps * input_imgs_grad_detached
                #     features_adv, logits_adv = self.model(input_imgs_adv)
                #     losst_adv, prect_adv, _ = self.criterion(logits_adv, targets)
                #     self.writer.add_scalar('vis/loss-adv', losst_adv.item(), self.iter)
                #     self.writer.add_scalar('vis/prec-adv', prect_adv.item(), self.iter)
                #     (self.args.adv_inp * losst_adv).backward()
                #     # (self.args.adv_inp * losst_adv + losst).backward()
                # optimizer.step()
                # method2
                optimizer.zero_grad()
                losst.backward()
                input_imgs_grad = input_imgs.grad.detach()
                # input_imgs.requires_grad_(False)
                if self.args.aux == 'l2_adv':
                    input_imgs_adv = input_imgs + self.args.adv_inp_eps * l2_normalize(input_imgs_grad)
                elif self.args.aux == 'linf_adv':
                    input_imgs_adv = input_imgs + self.args.adv_inp_eps * torch.sign(input_imgs_grad)
                else:
                    input_imgs_adv = input_imgs + self.args.adv_inp_eps * input_imgs_grad
                features_adv, logits_adv = self.model(input_imgs_adv)
                losst_adv, prect_adv, _ = self.criterion(logits_adv, targets)
                self.writer.add_scalar('vis/loss-adv', losst_adv.item(), self.iter)
                self.writer.add_scalar('vis/prec-adv', prect_adv.item(), self.iter)
                (self.args.adv_inp * losst_adv).backward()
                if math.isnan(losst_adv.item()):
                    print('fail')
                    raise ValueError('nan')
                optimizer.step()
            elif self.args.double != 0 and self.args.adv_inp == 0:
                optimizer.zero_grad()
                losst.backward(retain_graph=True)
                input_imgs_grad_attached = torch.autograd.grad(
                    outputs=losst, inputs=input_imgs,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0].view(
                    batch_size, -1
                )
                # grad_fea = [torch.autograd.grad(
                #     outputs=features[0][i],
                #     inputs=input_imgs,
                #     # create_graph=True,
                #     retain_graph=True,
                #     only_inputs=True,
                # )[0] for i in range(128)]
                if self.args.aux == 'l1_grad':
                    grad_reg = (input_imgs_grad_attached.norm(1, dim=1)).mean()
                else:
                    grad_reg = (input_imgs_grad_attached.norm(2, dim=1) ** 2).mean()
                if np.count_nonzero(self.args.reg_mid_fea) == 0:
                    (self.args.double * grad_reg).backward()
                else:
                    (self.args.double * grad_reg).backward(retain_graph=True)
                ind_max = np.nonzero(self.args.reg_mid_fea)[0].max()
                for ind, weight in enumerate(self.args.reg_mid_fea):
                    fea = mid_feas[ind]
                    if weight != 0:
                        reg = reg_mid_fea(fea, input_imgs, weight,
                                          retain_graph=(ind != ind_max))
                        self.writer.add_scalar(f'vis/contract_fea_{ind+1}',
                                               reg, self.iter)
                        if ind == ind_max: break
                self.writer.add_scalar('vis/grad_reg', grad_reg.item(), self.iter)
                optimizer.step()
            elif self.args.double != 0 and self.args.adv_inp != 0:
                optimizer.zero_grad()

                input_imgs_grad_attached = torch.autograd.grad(
                    outputs=losst, inputs=input_imgs,
                    create_graph=True, retain_graph=True,
                    only_inputs=True
                )[0]

                input_imgs_grad = input_imgs_grad_attached.detach()
                input_imgs_grad_attached = input_imgs_grad_attached.view(
                    input_imgs_grad_attached.size(0), -1
                )
                if 'l1_grad' in self.args.aux:
                    grad_reg = (input_imgs_grad_attached.norm(1, dim=1)).mean()
                else:
                    grad_reg = (input_imgs_grad_attached.norm(2, dim=1) ** 2).mean()
                (losst + self.args.double * grad_reg).backward()
                self.writer.add_scalar('vis/grad_reg', grad_reg.item(), self.iter)

                # input_imgs.requires_grad_(False)
                if 'l2_adv' in self.args.aux:
                    input_imgs_adv = input_imgs + self.args.adv_inp_eps * l2_normalize(input_imgs_grad)
                elif 'linf_adv' in self.args.aux:
                    input_imgs_adv = input_imgs + self.args.adv_inp_eps * torch.sign(input_imgs_grad)
                else:
                    input_imgs_adv = input_imgs + self.args.adv_inp_eps * input_imgs_grad
                features_adv, logits_adv = self.model(input_imgs_adv)
                losst_adv, _, _ = self.criterion(logits_adv, targets)
                (self.args.adv_inp * losst_adv).backward()
                self.writer.add_scalar('vis/loss_adv_inp', losst_adv.item(), self.iter)
                optimizer.step()
            elif self.args.adv_fea != 0:
                # method 1
                optimizer.zero_grad()
                features_grad = torch.autograd.grad(
                    outputs=losst, inputs=features,
                    create_graph=True, retain_graph=True,
                    only_inputs=True
                )[0].detach()
                # input_imgs.requires_grad = False
                assert features_grad.requires_grad is False
                if 'l2_adv' in self.args.aux:
                    features_advtrue = features + self.args.adv_inp_eps * l2_normalize(features_grad)
                elif 'linf_adv' in self.args.aux:
                    features_advtrue = features + self.args.adv_fea_eps * torch.sign(features_grad)
                else:
                    features_advtrue = features + self.args.adv_inp_eps * features_grad
                losst_advtrue, _, _ = self.criterion(features_advtrue, targets)
                (losst + self.args.adv_fea * losst_advtrue).backward()
                self.writer.add_scalar('vis/loss_adv_fea', losst_advtrue.item(), self.iter)
                optimizer.step()
                # method 2
            elif np.count_nonzero(self.args.reg_mid_fea) != 0:
                optimizer.zero_grad()
                losst.backward(retain_graph=True)
                ind_max = np.nonzero(self.args.reg_mid_fea)[0].max()
                for ind, weight in enumerate(self.args.reg_mid_fea):
                    fea = mid_feas[ind]
                    if weight != 0:
                        reg = reg_mid_fea(fea, input_imgs, weight,
                                          retain_graph=(ind != ind_max),
                                          # projed=x5_grad_reg,
                                          projed=None,
                                          )
                        self.writer.add_scalar(f'vis/contract_fea_{ind+1}', reg, self.iter)
                        if ind == ind_max: break
                optimizer.step()
            else:
                optimizer.zero_grad()
                losst.backward()

                # losst.backward(retain_graph=True)
                # input_imgs_grad_attached = torch.autograd.grad(
                #     outputs=losst, inputs=input_imgs,
                #     # create_graph=True,
                #     # retain_graph=False,
                #     only_inputs=True
                # )[0].view(
                #     batch_size, -1
                # )
                # grad_reg = (input_imgs_grad_attached.norm(2, dim=1) ** 2).mean()
                # self.writer.add_scalar('vis/grad_reg', grad_reg.item(), self.iter)

                optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print(f'Epoch: [{epoch}][{i+1}/{len(data_loader)}]  '
                      f'Time {batch_time.val:.1f}/{batch_time.avg:.1f}  '
                      f'Data {data_time.val:.1f}/{data_time.avg:.1f}  '
                      f'loss {losses.val:.2f}/{losses.avg:.2f}  '
                      f'prec {precisions.val:.2%}/{precisions.avg:.2%}  '
                      )
            # break
        return collections.OrderedDict({
            'ttl-time': batch_time.avg,
            'data-time': data_time.avg,
            'loss_tri': losses.avg,
            'prec_tri': precisions.avg,
        })


def l2_normalize(x):
    # can only handle (128,2048) or (128,2048,8,4)
    shape = x.size()
    x1 = x.view(shape[0], -1)
    x2 = x1 / x1.norm(p=2, dim=1, keepdim=True)
    return x2.view(shape)


def reg_mid_fea(fea, input_imgs, weight, retain_graph=True, projed=None):
    bs, fea_len = fea.size()
    if projed is None:
        unit_vec = torch.randn(fea_len).cuda()
        prj_fea = (fea * unit_vec).mean()
    else:
        prj_fea = projed
    fea_grad = torch.autograd.grad(
        outputs=prj_fea, inputs=input_imgs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0].view(bs, -1)
    features_grad_reg = (fea_grad.norm(2, dim=1) ** 2).mean()
    loss = features_grad_reg * weight
    loss.backward(retain_graph=retain_graph)
    return features_grad_reg.item()


'''
class TCXTrainer(object):
    def __init__(self, model, criterion, logs_at='work/vis', dbg=False, args=None, dop_info=None, **kwargs):
        self.model = model
        self.crit_tri = criterion[0]
        self.crit_cent = criterion[1]
        self.crit_xent = criterion[2]
        self.iter = 0
        self.dbg = dbg
        self.cls_weight = args.cls_weight
        self.tri_weight = args.tri_weight
        self.weight_cent = args.weight_cent
        self.args = args
        self.dop_info = dop_info
        if dbg:
            mkdir_p(logs_at, delete=True)
            self.writer = SummaryWriter(logs_at)
        else:
            self.writer = None

    def _forward(self, inputs, targets, cids=None):
        _zero = torch.zeros(1).cuda()

        out_embed, out_cls = self.model(*inputs)
        # logging.info('output embedding {} outpus class{}'.format(out_embed.size(), out_cls.size()))
        if self.dbg and self.iter % 1000 == 0:
            self.writer.add_histogram('1_input', inputs[0], self.iter)
            self.writer.add_histogram('2_feature', out_embed, self.iter)
            x = vutils.make_grid(
                to_torch(inputs[0]), normalize=True, scale_each=True)
            self.writer.add_image('input', x, self.iter)
        # triplet
        if not math.isclose(self.tri_weight, 0):
            if self.dbg and self.iter % 100 == 0:
                losst, prect, dist, dist_ap, dist_an = self.crit_tri(
                    out_embed, targets, dbg=self.dbg, cids=cids)
                diff = dist_an - dist_ap
                self.writer.add_histogram('an-ap', diff, self.iter)
                self.writer.add_histogram('dist', dist, self.iter)
                self.writer.add_histogram('ap', dist_ap, self.iter)
                self.writer.add_histogram('an', dist_an, self.iter)
            else:
                losst, prect, dist = self.crit_tri(
                    out_embed, targets, dbg=False, cids=cids)
        else:
            losst, prect, dist = _zero, _zero, _zero
        # xent
        if not math.isclose(self.cls_weight, 0):
            lossx = self.crit_xent(out_cls, targets)
            precx, = accuracy(out_cls.data, targets.data)
            precx = precx[0]
        else:
            lossx, precx = _zero, _zero
        # cent
        if not math.isclose(self.args.weight_cent, 0):
            loss_cent, loss_dis, distmat_cent, cent_pull = self.crit_cent(
                out_embed, targets, )
        else:
            loss_cent, loss_dis, distmat_cent, cent_pull = _zero, _zero, _zero, _zero

        if not math.isclose(self.weight_cent, 0):
            update_dop_center(distmat_cent, self.dop_info)
        elif not math.isclose(self.tri_weight, 0):
            update_dop_tri(dist, targets, self.dop_info)

        self.iter += 1
        loss_comb = self.tri_weight * losst + \
                    self.weight_cent * loss_cent + \
                    self.args.weight_dis_cent * loss_dis + \
                    self.args.cls_weight * lossx
        # gradients = torch.autograd.grad(
        #     outputs=losst, inputs=inputs[0],
        #     create_graph=True, retain_graph=True, only_inputs=True
        # )[0]
        # ...
        # loss_comb += (gradients.norm(2, dim=1)**2).mean()

        # noise = 0.3 * gradients.detach()
        # ...
        # input2 = inputs[0] + noise

        # out_embed2, out_cls2 = self.model(input2)
        # losst2, prect2, dist2 = self.crit_tri(out_embed2, targets, dbg=False, cids=cids)
        # loss_comb = (loss_comb + losst2) / 2.

        logging.debug(
            f'tri loss {losst.item()}; '
            f' loss_cent is {loss_cent.item()}; '
            f' loss_dis is {loss_dis.item()}')
        if loss_comb > 1e8:
            raise ValueError('loss too large')
        elif math.isnan(loss_comb.data.cpu()):
            raise ValueError(f'loss nan {loss_comb}')
        if self.dbg and self.iter % 1 == 0:
            self.writer.add_scalar('vis/prec-triplet', prect.item(), self.iter)
            self.writer.add_scalar('vis/lr', self.lr, self.iter)
            self.writer.add_scalar('vis/loss-center', loss_cent.item(), self.iter)
            self.writer.add_scalar('vis/loss-center-dis', loss_dis.item(), self.iter)
            self.writer.add_scalar('vis/loss-triplet', losst.item(), self.iter)
            self.writer.add_scalar('vis/center-pull', cent_pull.item(), self.iter)
            self.writer.add_scalar('vis/loss-softmax', lossx.item(), self.iter)
            self.writer.add_scalar('vis/prec-softmax', precx.item(), self.iter)
        return loss_comb, prect, precx

    def train(self, epoch, data_loader, optimizer, optimizer_cent=None, print_freq=5, schedule=None):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions_tri = AverageMeter()
        precisionsx = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            input_imgs = inputs.get('img').cuda()
            input_imgs.requires_grad = True
            targets = inputs.get('pid').cuda()

            if schedule is not None:
                schedule.batch_step()
            self.lr = optimizer.param_groups[0]['lr']
            self.model.train()
            # todo for tcx loss
            loss_comb, prect, precx = self._forward(inputs, targets, cids)
            if isinstance(targets, tuple):
                targets, _ = targets
            losses.update(to_numpy(loss_comb), targets.size(0))
            precisions_tri.update(to_numpy(prect), targets.size(0))
            precisionsx.update(to_numpy(precx), targets.size(0))
            optimizer_cent.zero_grad()
            optimizer.zero_grad()
            loss_comb.backward()
            # todo version2 adv-train
            optimizer.step()
            # for param in self.criterion2.parameters():
            for name, param in self.crit_cent.named_parameters():
                if name != 'centers':
                    continue
                # print(name)
                if not math.isclose(self.weight_cent, 0.):
                    param.grad.data *= (1. / self.weight_cent)
            optimizer_cent.step()
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print(f'Epoch: [{epoch}][{i+1}/{len(data_loader)}]  '
                      f'Time {batch_time.val:.1f}/{batch_time.avg:.1f}  '
                      f'Data {data_time.val:.1f}/{data_time.avg:.1f}  '
                      f'loss {losses.val:.2e}/{losses.avg:.2e}  '
                      f'prect {precisions_tri.val:.2%}/{precisions_tri.avg:.2%}  '
                      f'precx {precisionsx.val:.2%}/{precisionsx.avg:.2%}  '
                      )
            # break
        return collections.OrderedDict({
            'ttl-time': batch_time.avg,
            'data-time': data_time.avg,
            'loss_tri': losses.avg,
            'prec_tri': precisions_tri.avg,
            'prec_xent': precisionsx.avg
        })


class XentTriTrainer(object):
    def __init__(self, model, criterion, logs_at='work/vis', dbg=False, args=None, dop_info=None, **kwargs):
        self.model = model
        self.criterion = criterion[0]
        self.criterion2 = criterion[1]
        self.iter = 0
        self.dbg = dbg
        self.cls_weight = args.cls_weight
        self.tri_weight = args.tri_weight
        self.dop_info = dop_info
        if dbg:
            mkdir_p(logs_at, delete=True)
            self.writer = SummaryWriter(logs_at)
        else:
            self.writer = None

    def _parse_data(self, inputs):
        imgs, npys, fnames, pids = inputs.get('img'), inputs.get(
            'npy'), inputs.get('fname'), inputs.get('pid')
        cids = inputs.get('cid')
        inputs = [imgs, npys]
        inputs = to_variable(inputs, requires_grad=False)
        targets = to_variable(pids, requires_grad=False)
        cids = to_variable(cids, requires_grad=False)
        return inputs, targets, fnames, cids

    def _forward(self, inputs, targets, cids=None):
        outputs, outputs2 = self.model(*inputs)
        # logging.info('{} {}'.format(outputs.size(), outputs2.size()))
        if self.dbg and self.iter % 1000 == 0:
            self.writer.add_histogram('1_input', inputs[0], self.iter)
            self.writer.add_histogram('2_feature', outputs, self.iter)
            x = vutils.make_grid(
                to_torch(inputs[0]), normalize=True, scale_each=True)
            self.writer.add_image('input', x, self.iter)

        loss2 = self.criterion2(outputs2, targets)  # criterion2 is xent
        prec2, = accuracy(outputs2.data, targets.data)
        prec2 = prec2[0]
        if self.dbg and self.iter % 100 == 0:
            loss, prec, dist, dist_ap, dist_an = self.criterion(
                outputs, targets, dbg=self.dbg, cids=cids)
            diff = dist_an - dist_ap
            self.writer.add_histogram('an-ap', diff, self.iter)
            self.writer.add_histogram('dist', dist, self.iter)
            self.writer.add_histogram('ap', dist_ap, self.iter)
            self.writer.add_histogram('an', dist_an, self.iter)
        else:
            loss, prec, dist = self.criterion(
                outputs, targets, dbg=False, cids=cids)  # criterion1 is triplet
        if self.tri_weight != 0:
            update_dop_tri(dist, targets, self.dop_info)

        self.iter += 1
        loss_comb = self.tri_weight * loss + self.cls_weight * loss2

        if self.dbg and self.iter % 10 == 0:
            self.writer.add_scalar('vis/prec-triplet', prec, self.iter)
            self.writer.add_scalar('vis/loss-triplet', loss, self.iter)
            # self.writer.add_scalar('vis/lr',lr )
            self.writer.add_scalar('vis/loss-softmax', loss2, self.iter)
            self.writer.add_scalar('vis/prec-softmax', prec2, self.iter)
            self.writer.add_scalar('vis/loss-ttl', loss_comb, self.iter)

        return loss_comb, loss, loss2, prec, prec2

    def train(self, epoch, data_loader, optimizer, print_freq=5, schedule=None):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses2 = AverageMeter()
        precisions = AverageMeter()
        precisions2 = AverageMeter()

        end = time.time()

        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            inputs, targets, fnames, cids = self._parse_data(inputs)
            if schedule is not None:
                schedule.batch_step()
            self.lr = optimizer.param_groups[0]['lr']
            self.model.train()
            loss_comb, loss, loss2, prec1, prec2 = self._forward(
                inputs, targets, cids)
            if isinstance(targets, tuple):
                targets, _ = targets
            losses.update(to_numpy(loss), targets.size(0))
            losses2.update(to_numpy(loss2), targets.size(0))
            precisions.update(to_numpy(prec1), targets.size(0))
            precisions2.update(to_numpy(prec2), targets.size(0))

            optimizer.zero_grad()
            loss_comb.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print(f'Epoch: [{epoch}][{i+1}/{len(data_loader)}]  '
                      f'Time {batch_time.val:.1f}/{batch_time.avg:.1f}  '
                      f'Data {data_time.val:.1f}/{data_time.avg:.1f}  '
                      f'loss {losses.val:.2f}/{losses.avg:.2f}  '
                      f'loss_cls {losses2.val:.2f}/{losses2.avg:.2f}  '
                      f'prec {precisions.val:.2%}/{precisions.avg:.2%}  '
                      f'prec_cls {precisions2.val:.2%}/{precisions2.avg:.2%}  '
                      )
            # break
        return collections.OrderedDict({
            'ttl-time': batch_time.avg,
            'data-time': data_time.avg,
            'loss_tri': losses.avg,
            'prec_tri': precisions.avg,
        })


class TriCenterTrainer(object):
    def __init__(self, model, criterion, logs_at='work/vis', dbg=False, args=None, dop_info=None, **kwargs):
        self.model = model
        self.criterion = criterion[0]
        self.criterion2 = criterion[1]
        self.iter = 0
        self.dbg = dbg
        self.cls_weight = args.cls_weight
        self.tri_weight = args.tri_weight
        self.weight_cent = args.weight_cent
        self.args = args
        self.dop_info = dop_info
        if dbg:
            mkdir_p(logs_at, delete=True)
            self.writer = SummaryWriter(logs_at)
        else:
            self.writer = None

    def _parse_data(self, inputs):
        imgs, npys, fnames, pids = inputs.get('img'), inputs.get(
            'npy'), inputs.get('fname'), inputs.get('pid')
        cids = inputs.get('cid')
        inputs = [imgs, npys]
        inputs = to_variable(inputs, requires_grad=False)
        targets = to_variable(pids, requires_grad=False)
        cids = to_variable(cids, requires_grad=False)
        return inputs, targets, fnames, cids

    def _forward(self, inputs, targets, cids=None):
        outputs, outputs2 = self.model(*inputs)
        # logging.info('{} {}'.format(outputs.size(), outputs2.size()))
        if self.dbg and self.iter % 1000 == 0:
            self.writer.add_histogram('1_input', inputs[0], self.iter)
            self.writer.add_histogram('2_feature', outputs, self.iter)
            x = vutils.make_grid(
                to_torch(inputs[0]), normalize=True, scale_each=True)
            self.writer.add_image('input', x, self.iter)

        if self.dbg and self.iter % 100 == 0:
            loss, prec, dist, dist_ap, dist_an = self.criterion(
                outputs, targets, dbg=self.dbg, cids=cids)
            diff = dist_an - dist_ap
            self.writer.add_histogram('an-ap', diff, self.iter)
            self.writer.add_histogram('dist', dist, self.iter)
            self.writer.add_histogram('ap', dist_ap, self.iter)
            self.writer.add_histogram('an', dist_an, self.iter)
        else:
            loss, prec, dist = self.criterion(
                outputs, targets, dbg=False, cids=cids)

        loss_cent, loss_dis, distmat_cent, loss_cent_pull = self.criterion2(
            outputs, targets, )
        # update_dop_tri(dist, targets, self.dop_info)
        update_dop_center(distmat_cent, self.dop_info)

        self.iter += 1
        if self.args.weight_lda is None:
            loss_comb = self.tri_weight * loss + self.weight_cent * \
                        loss_cent + self.args.weight_dis_cent * loss_dis
        else:
            loss_comb = loss + self.args.weight_lda * loss_cent / (-loss_dis)

        logging.debug(
            f'tri loss {loss.item()}; loss_cent is {loss_cent.item()};  loss_dis is {loss_dis.item()}')
        if loss_comb > 1e8:
            raise ValueError('loss too large')

        if self.dbg and self.iter % 10 == 0:
            self.writer.add_scalar('vis/prec-triplet', prec, self.iter)
            self.writer.add_scalar('vis/lr', self.lr, self.iter)
            self.writer.add_scalar('vis/loss-center', loss_cent, self.iter)
            self.writer.add_scalar('vis/loss-center-dis', loss_dis, self.iter)
            self.writer.add_scalar('vis/loss-triplet', loss, self.iter)
            self.writer.add_scalar(
                'vis/loss-center-pull', loss_cent_pull, self.iter)

        return loss_comb, prec

    def train(self, epoch, data_loader, optimizer, optimizer_cent=None, print_freq=5, schedule=None):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            inputs, targets, fnames, cids = self._parse_data(inputs)
            if schedule is not None:
                schedule.batch_step()
            self.lr = optimizer.param_groups[0]['lr']
            self.model.train()
            loss_comb, prec = self._forward(inputs, targets, cids)
            if isinstance(targets, tuple):
                targets, _ = targets
            losses.update(to_numpy(loss_comb), targets.size(0))
            precisions.update(to_numpy(prec), targets.size(0))
            optimizer_cent.zero_grad()
            optimizer.zero_grad()
            loss_comb.backward()

            optimizer.step()
            # for param in self.criterion2.parameters():
            for name, param in self.criterion2.named_parameters():
                if name != 'centers':
                    continue
                # print(name)
                if not math.isclose(self.weight_cent, 0.):
                    param.grad.data *= (1. / self.weight_cent)
            optimizer_cent.step()
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print(f'Epoch: [{epoch}][{i+1}/{len(data_loader)}]  '
                      f'Time {batch_time.val:.1f}/{batch_time.avg:.1f}  '
                      f'Data {data_time.val:.1f}/{data_time.avg:.1f}  '
                      f'loss {losses.val:.2f}/{losses.avg:.2f}  '
                      f'prec {precisions.val:.2%}/{precisions.avg:.2%}  '
                      )
            # break
            # exit(-1)
        return collections.OrderedDict({
            'ttl-time': batch_time.avg,
            'data-time': data_time.avg,
            'loss_tri': losses.avg,
            'prec_tri': precisions.avg,
        })
'''


class XentTrainer(object):
    def __init__(self, model, criterion, logs_at='work/vis', dbg=True, args=None, **kwargs):
        self.model = model
        self.criterion = criterion[0]
        self.iter = 0
        self.dbg = dbg
        if dbg:
            mkdir_p(logs_at, delete=True)
            self.writer = SummaryWriter(logs_at)
        else:
            self.writer = None
        self.args = args

    def train(self, epoch, data_loader, optimizer, print_freq=5, schedule=None):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            input_imgs = inputs.get('img').cuda()

            if self.dbg % 100 == 0:
                x = vutils.make_grid(
                    input_imgs, normalize=True, scale_each=True)
                self.writer.add_image('input', x, self.iter)
            input_imgs.requires_grad = True
            targets = inputs.get('pid').cuda()
            if schedule is not None:
                schedule.batch_step()
            self.lr = optimizer.param_groups[0]['lr']
            self.model.train()
            features, logits = self.model(input_imgs)
            loss = self.criterion(logits, targets)
            # loss is softmax loss
            prec = accuracy(logits, targets.data)[0]
            if isinstance(targets, tuple):
                targets, _ = targets
            losses.update(loss.item(), targets.size(0))
            precisions.update(prec.item(), targets.size(0))
            if self.dbg and self.iter % 10 == 0:
                self.writer.add_scalar('vis/prec-softmax', prec.item(), self.iter)
                self.writer.add_scalar('vis/loss-softmax', loss.item(), self.iter)

            if self.args.adv_inp != 0 and self.args.double == 0:
                optimizer.zero_grad()
                loss.backward()
                input_imgs_grad = input_imgs.grad.detach()
                # input_imgs.requires_grad_(False)
                if self.args.aux == 'l2_adv':
                    input_imgs_adv = input_imgs + self.args.adv_inp_eps * l2_normalize(input_imgs_grad)
                elif self.args.aux == 'linf_adv':
                    input_imgs_adv = input_imgs + self.args.adv_inp_eps * torch.sign(input_imgs_grad)
                else:
                    input_imgs_adv = input_imgs + self.args.adv_inp_eps * input_imgs_grad
                features_adv, logits_adv = self.model(input_imgs_adv)
                losst_adv = self.criterion(logits_adv, targets)
                self.writer.add_scalar('vis/loss-adv')
                (self.args.adv_inp * losst_adv).backward()
                optimizer.step()
            elif self.args.double != 0 and self.args.adv_inp == 0:
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                input_imgs_grad_attached = torch.autograd.grad(
                    outputs=loss, inputs=input_imgs,
                    create_graph=True, retain_graph=True,
                    only_inputs=True
                )[0]
                input_imgs_grad_attached = input_imgs_grad_attached.view(
                    input_imgs_grad_attached.size(0), -1
                )
                if self.args.aux == 'l1_grad':
                    grad_reg = (input_imgs_grad_attached.norm(1, dim=1)).mean()
                else:
                    grad_reg = (input_imgs_grad_attached.norm(2, dim=1) ** 2).mean()
                (self.args.double * grad_reg).backward()
                self.writer.add_scalar('vis/grad_reg', grad_reg.item(), self.iter)
                optimizer.step()
            elif self.args.double != 0 and self.args.adv_inp != 0:
                optimizer.zero_grad()

                input_imgs_grad_attached = torch.autograd.grad(
                    outputs=loss, inputs=input_imgs,
                    create_graph=True, retain_graph=True,
                    only_inputs=True
                )[0]

                input_imgs_grad = input_imgs_grad_attached.detach()
                input_imgs_grad_attached = input_imgs_grad_attached.view(
                    input_imgs_grad_attached.size(0), -1
                )
                if 'l1_grad' in self.args.aux:
                    grad_reg = (input_imgs_grad_attached.norm(1, dim=1)).mean()
                else:
                    grad_reg = (input_imgs_grad_attached.norm(2, dim=1) ** 2).mean()
                (loss + self.args.double * grad_reg).backward()
                self.writer.add_scalar('vis/grad_reg', grad_reg.item(), self.iter)

                # input_imgs.requires_grad_(False)
                if 'l2_adv' in self.args.aux:
                    input_imgs_adv = input_imgs + self.args.adv_inp_eps * l2_normalize(input_imgs_grad)
                elif 'linf_adv' in self.args.aux:
                    input_imgs_adv = input_imgs + self.args.adv_inp_eps * torch.sign(input_imgs_grad)
                else:
                    input_imgs_adv = input_imgs + self.args.adv_inp_eps * input_imgs_grad
                features_adv, logits_adv = self.model(input_imgs_adv)
                losst_adv = self.criterion(logits_adv, targets)
                (self.args.adv_inp * losst_adv).backward()
                self.writer.add_scalar('vis/loss_adv_inp', losst_adv.item(), self.iter)
                optimizer.step()
            elif self.args.adv_fea != 0:
                # method 1
                optimizer.zero_grad()
                features_grad = torch.autograd.grad(
                    outputs=loss, inputs=features,
                    create_graph=True, retain_graph=True,
                    only_inputs=True
                )[0].detach()
                # input_imgs.requires_grad = False
                assert features_grad.requires_grad is False
                features_advtrue = features + self.args.adv_fea_eps * torch.sign(features_grad)
                losst_advtrue, _, _ = self.criterion(features_advtrue, targets)
                (loss + self.args.adv_fea * losst_advtrue).backard()
                self.writer.add_scalar('vis/loss_adv_fea', losst_advtrue.item(), self.iter)
                optimizer.step()
                # method 2
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print(f'Epoch: [{epoch}][{i+1}/{len(data_loader)}]  '
                      f'Time {batch_time.val:.1f}/{batch_time.avg:.1f}  '
                      f'Data {data_time.val:.1f}/{data_time.avg:.1f}  '
                      f'loss {losses.val:.2f}/{losses.avg:.2f}  '
                      f'prec {precisions.val:.2%}/{precisions.avg:.2%}  '
                      )
            # break
        return collections.OrderedDict({
            'ttl-time': batch_time.avg,
            'data-time': data_time.avg,
            'loss': losses.avg,
            'prec': precisions.avg,
        })
