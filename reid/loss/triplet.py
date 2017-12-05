from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import subprocess
import numpy as np, numpy


class TripletLoss(nn.Module):
    def __init__(self, margin=0, mode='hard'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mode = mode

    def forward(self, inputs, targets, dbg=False):
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
                posp = dist[i][mask[i]]
                _, posp_ind = posp.max(0)
                posp_ind = Variable(posp_ind.data, requires_grad=False)
                posp_max = posp[posp_ind]
                dist_ap.append(posp_max)  # dist[i][mask[i]].max()

                negp = dist[i][mask[i] == 0]
                _, negp_ind = negp.min(0)
                negp_ind = Variable(negp_ind.data, requires_grad=False)
                negp_min = negp[negp_ind]

                dist_an.append(negp_min)  # dist[i][mask[i] == 0].min()

            elif self.mode == 'rand':
                posp = dist[i][mask[i]]
                dist_ap.append(posp[numpy.random.randint(0, posp.size(0))])

                negp = dist[i][mask[i] == 0]
                dist_an.append(negp[numpy.random.randint(0, negp.size(0))])
            # todo check
            elif self.mode == 'lift':
                negp = dist[i][mask[i] == 0]
                posp = (dist[i][mask[i]].max()).expand(negp.size(0))

                dist_ap.extend(posp)
                dist_an.extend(negp)
            elif self.mode == 'all':
                posp = dist[i][mask[i]]
                negp = dist[i][mask[i] == 0]
                np_, nn = posp.size(0), negp.size(0)
                posp = posp.expand((nn, np_)).t()
                negp = negp.expand((np_, nn))  # .contiguous().view(-1)
                for posp_, negp_ in zip(posp, negp):
                    dist_ap.extend(posp_)
                    dist_an.extend(negp_)

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)

        if not dbg:
            return loss, prec
        else:
            return loss, prec, dist, dist_ap, dist_an


def stat(tensor):
    return tensor.min(), tensor.mean(), tensor.max(), tensor.std(), tensor.size()
