from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class TupletLoss(nn.Module):
    def __init__(self, margin=0,loss='hinge'):
        super(TupletLoss, self).__init__()
        self.margin = margin
        self.loss=loss
        if loss == 'hinge':
            self.loss_f = nn.HingeEmbeddingLoss(margin)
        elif loss =='bce':
            self.loss_f=nn.BCELoss()

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        pairs = torch.cat(dist_ap + dist_an)
        if self.loss=='hinge':
            # Compute bce loss
            y = torch.from_numpy(np.concatenate((
                np.ones((n,)),
                np.ones((n,)) * -1
            )))
            y = y.type_as(pairs.data)
            y.resize_as_(pairs.data)
            y = Variable(y)
            loss = self.loss_f(input=pairs, target=y)
            prec = ((pairs.data < self.margin)
                    == (y.data > 0)).sum() * 1. / y.size(0)
            return loss, prec
        elif self.loss == 'bce':
            pairs = torch.exp(-pairs)
            # Compute bce loss
            y = torch.from_numpy(np.concatenate((
                np.ones((n,)),
                np.zeros((n,))
            )))
            y = y.type_as(pairs.data)
            y.resize_as_(pairs.data)
            y = Variable(y)
            loss = self.bce_with_logit(input=pairs, target=y)
            prec = ((pairs.data > 0.5).type_as(y.data) == y.data).sum() * 1. / y.size(0)
            return loss, prec
