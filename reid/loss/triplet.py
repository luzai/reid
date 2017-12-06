from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import subprocess
import numpy as np, numpy
import lz


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
        all_ind = Variable(torch.arange(0, n).type(torch.LongTensor), requires_grad=False, volatile=True).cuda()
        posp_inds, negp_inds = [], []

        for i in range(n):
            if self.mode == 'hard':
                some_ind = all_ind[mask[i]]
                some_pos = dist[i][mask[i]]
                _, ind4some_ind = some_pos.max(0)
                global_ind = some_ind[ind4some_ind]
                global_ind = Variable(global_ind.data, requires_grad=False)
                posp_inds.append(global_ind)
                dist_ap.append(some_pos[ind4some_ind])

                some_ind  = all_ind[mask[i] == 0]
                some_neg = dist[i][mask[i]==0]
                _, ind4some_ind= some_neg.min(0)
                global_ind = some_ind[ind4some_ind]
                global_ind = Variable(global_ind.data, requires_grad=False)
                negp_inds.append(global_ind)
                dist_an.append(some_neg[ind4some_ind])  # dist[i][mask[i] == 0].min()

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

        posp_inds, negp_inds = torch.cat(posp_inds), torch.cat(negp_inds)
        replay_ind = get_replay_ind(posp_inds, negp_inds, dist_an - dist_ap)
        if not dbg:
            return loss, prec, replay_ind
        else:
            return loss, prec, replay_ind,  dist, dist_ap, dist_an


def get_replay_ind(posp_inds, negp_inds, diff):
    batch_size=diff.size(0)
    pinds = lz.to_numpy(posp_inds)
    ninds = lz.to_numpy(negp_inds)
    diff = lz.to_numpy(diff)

    # db = lz.Database('tmp.h5', 'w')
    # db['pinds'] = pinds
    # db['ninds'] = ninds
    # db['diff'] = diff
    # db.close()

    thresh1 = np.percentile(diff, 45)
    thresh2 = np.percentile(diff, 95)

    sel_ind = np.nonzero(np.logical_and(diff > thresh1, diff < thresh2))[0]
    sel=np.concatenate(
        (pinds[sel_ind], ninds[sel_ind], sel_ind)
    )
    bins, _ = np.histogram(sel, bins=batch_size, range=(0, batch_size))
    bins = bins / bins.sum()
    return bins

