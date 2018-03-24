from lz import *
import lz

from tensorboardX import SummaryWriter
import numpy as np, numpy


def select(dist, range, descend=True, return_ind=False, global_ind=None):
    dist, ind = torch.sort(dist, descending=descend)
    if not return_ind:
        return dist[range[0]:range[1]]
    else:
        return global_ind[dist[range[0]:range[1]]], global_ind[ind[range[0]:range[1]]]


class QuadLoss(nn.Module):
    def __init__(self, margin=0, mode='hard', **kwargs):
        super(QuadLoss, self).__init__()
        self.margin = margin
        self.mode = mode
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, dbg=False):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        dist_ap, dist_an, dist_n12 = [], [], []
        for i in range(n):
            some_pos = dist[i][mask[i]]
            some_neg = dist[i][mask[i] == 0]

            neg, n1_ind = some_neg.min(0)
            pos = some_pos.max()

            dist_ap.append(pos)
            dist_an.append(neg)

            n1_ind = to_numpy(n1_ind)[0]
            some_n2 = dist[n1_ind][mask[n1_ind] == 0]
            n2 = some_n2.min()
            dist_n12.append(n2)

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        dist_n12 = torch.cat(dist_n12)
        if torch.cuda.is_available():
            y = Variable(to_torch(np.ones(dist_an.size())).type(torch.cuda.FloatTensor), requires_grad=False).cuda()
        else:
            y = Variable(to_torch(np.ones(dist_an.size())).type(torch.FloatTensor), requires_grad=False)

        loss = self.ranking_loss(dist_an, dist_ap, y) + 0.1 * self.ranking_loss(dist_n12, dist_ap, y)
        # todo 0.1 and different margin
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)

        if not dbg:
            return loss, prec, dist
        else:
            return loss, prec, dist, dist_ap, dist_an


class QuinLoss(nn.Module):
    def __init__(self, margin=0, mode='hard', **kwargs):
        super(QuinLoss, self).__init__()
        self.margin = margin
        self.mode = mode
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, dbg=False):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        dist_ap, dist_an, dist_n12 = [], [], []
        for i in range(n):
            some_pos = dist[i][mask[i]]
            some_neg = dist[i][mask[i] == 0]

            neg, n1_ind = some_neg.min(0)
            pos = some_pos.max()

            dist_ap.append(pos)
            dist_an.append(neg)

            n1_ind = to_numpy(n1_ind)[0]
            some_n2 = dist[n1_ind][mask[n1_ind] == 0]
            n2 = some_n2.min()
            dist_n12.append(n2)

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        dist_n12 = torch.cat(dist_n12)
        if torch.cuda.is_available():
            y = Variable(to_torch(np.ones(dist_an.size())).type(torch.cuda.FloatTensor), requires_grad=False).cuda()
        else:
            y = Variable(to_torch(np.ones(dist_an.size())).type(torch.FloatTensor), requires_grad=False)

        loss = self.ranking_loss(dist_an, dist_ap, y) + 0.1 * self.ranking_loss(dist_n12, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)

        if not dbg:
            return loss, prec, dist
        else:
            return loss, prec, dist, dist_ap, dist_an


class TripletLoss(nn.Module):
    def __init__(self, margin=0, mode='hard'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mode = mode

    def forward(self, inputs, targets, dbg=False, cids=None):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        # all_ind = to_variable(torch.arange(0, n).type(torch.LongTensor), requires_grad=False, volatile=True)
        # posp_inds, negp_inds = [], []

        for i in range(n):
            if self.mode == 'hard':
                some_pos = dist[i][mask[i]]
                some_neg = dist[i][mask[i] == 0]
                # print(some_pos.size(),some_neg.size())

                # for pos in select(some_pos, (0, 1), descend=True):
                #     for neg in select(some_neg, (0, 1), descend=False):
                #         dist_ap.append(pos)
                #         dist_an.append(neg)

                neg = some_neg.min()
                pos = some_pos.max()

                dist_ap.append(pos)
                dist_an.append(neg)

            elif self.mode == 'pos.moderate':
                some_neg = dist[i][mask[i] == 0]
                neg = some_neg.min()
                some_pos = dist[i][mask[i]]
                some_pos_moderate = some_pos[(some_pos < neg).data]
                if not some_pos_moderate.shape:
                    pos = some_pos.min()
                else:
                    pos = some_pos_moderate.max()
                dist_ap.append(pos)
                dist_an.append(neg)
            elif self.mode == 'rand':
                posp = dist[i][mask[i]]
                dist_ap.append(posp[numpy.random.randint(0, posp.size(0))])

                negp = dist[i][mask[i] == 0]
                dist_an.append(negp[numpy.random.randint(0, negp.size(0))])
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
        # y = dist_an.data.new()
        # y.resize_as_(dist_an.data)
        # y.fill_(1)
        if torch.cuda.is_available():
            y = Variable(to_torch(np.ones(dist_an.size())).type(torch.cuda.FloatTensor), requires_grad=False).cuda()
        else:
            y = Variable(to_torch(np.ones(dist_an.size())).type(torch.FloatTensor), requires_grad=False)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)

        if not dbg:
            return loss, prec, dist
        else:
            return loss, prec, dist, dist_ap, dist_an


def get_replay_ind(posp_inds, negp_inds, diff):
    batch_size = diff.size(0)
    pinds = lz.to_numpy(posp_inds)
    ninds = lz.to_numpy(negp_inds)
    diff = lz.to_numpy(diff)

    # db = lz.Database('tmp.h5', 'w')
    # db['pinds'] = pinds
    # db['ninds'] = ninds
    # db['diff'] = diff
    # db.close()

    thresh1 = np.percentile(diff, 1)
    thresh2 = np.percentile(diff, 99)

    sel_ind = np.nonzero(np.logical_and(diff > thresh1, diff < thresh2))[0]
    sel = np.concatenate(
        (pinds[sel_ind], ninds[sel_ind], sel_ind)
    )
    bins, _ = np.histogram(sel, bins=batch_size, range=(0, batch_size))
    bins = bins / bins.sum()
    return bins
