from __future__ import print_function, absolute_import
import time, collections

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss, TupletLoss
from .utils.meters import AverageMeter
from lz import *


class BaseTrainer(object):
    def __init__(self, model, criterion, freeze=''):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.freeze = freeze

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
        imgs, fnames, pids, _ = inputs
        inputs = [Variable(imgs.cuda(), requires_grad=False)]

        targets = Variable(pids.cuda(), requires_grad=False)
        return inputs, targets

    def _forward(self, inputs, targets):
        # self.model.eval()
        if self.freeze == 'embed':
            # self.model.module.base_model.eval()
            self.model.module.embed_model.eval()
        pred, y = self.model(inputs[0], targets)
        loss = self.criterion(pred, y)
        prec1, = accuracy(pred.data, y.data)
        # ((pred.data[:,1]>pred.data[:,0]).type_as(y.data) == y.data).cpu().numpy()
        # (pred.data[:, 1] > pred.data[:, 0]).type_as(y.data).cpu().numpy()
        return loss, prec1[0]


def stat(tensor):
    return tensor.min(), tensor.mean(), tensor.max(), tensor.std(), tensor.size()


class SiameseTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        (imgs1, _, pids1, _), (imgs2, _, pids2, _) = inputs
        inputs = [Variable(imgs1), Variable(imgs2)]
        targets = Variable((pids1 == pids2).long().cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        loss = self.criterion(outputs, targets)
        prec1, = accuracy(outputs.data, targets.data)
        return loss, prec1[0]


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs.cuda(), requires_grad=False)]
        targets = Variable(pids.cuda(), requires_grad=False)
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        elif isinstance(self.criterion, TupletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec
