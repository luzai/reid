from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class MatrixLoss(nn.Module):
    def __init__(self, margin=0, mode='hard'):
        super(MatrixLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mode = mode

    def forward(self, dist, mask):
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
            if self.mode == 'hard':
                dist_ap.append(dist[i][mask[i]].max())
                dist_an.append(dist[i][mask[i] == 0].min())
            elif self.mode == 'rand':
                posp = dist[i][mask[i]]
                dist_ap.append(posp[np.random.randint(0, posp.size(0))])
                negp = dist[i][mask[i] == 0]
                dist_an.append(negp[np.random.randint(0, negp.size(0))])

                # posp.size(0)
                # negp.size(0)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec
