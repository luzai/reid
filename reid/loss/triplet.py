from lz import *
import lz

from tensorboardX import SummaryWriter
import numpy as np


def select(dist, range, descend=True, return_ind=False, global_ind=None):
    dist, ind = torch.sort(dist, descending=descend)
    if not return_ind:
        return dist[range[0]:range[1]]
    else:
        return global_ind[dist[range[0]:range[1]]], global_ind[ind[range[0]:range[1]]]


def calc_distmat2(x, y):
    num_x = x.size(0)
    num_y = y.size(0)
    distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(num_x, num_y) + \
              torch.pow(y, 2).sum(dim=1, keepdim=True).expand(num_y, num_x).t()
    distmat.addmm_(1, -2, x, y.t()).clamp_(min=1e-12, max=1e12)

    return distmat


def calc_distmat(x, y):
    num_x = x.size(0)
    num_y = y.size(0)
    distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(num_x, num_y) + \
              torch.pow(y, 2).sum(dim=1, keepdim=True).expand(num_y, num_x).t()
    distmat.addmm_(1, -2, x, y.t()).clamp_(min=1e-12, max=1e12).sqrt_()

    return distmat


from torch.nn import init


class CenterLoss(nn.Module):
    name = 'center'

    def __init__(self, num_classes, feat_dim,
                 margin2, margin3,
                 use_gpu=True, mode=None, push_scale=1.,
                 args=None, **kwargs):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.margin2 = margin2
        self.margin3 = margin3
        self.mode = mode
        self.args = args
        user_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(
                self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.feat_dim))
        init.kaiming_normal_(self.centers, mode='fan_out')

        # self.fc = nn.Linear(self.num_classes, self.num_classes - 1)
        # init.constant_(self.fc.bias, 1 )
        self.push_scale = push_scale
        self.push_wei = torch.ones(self.num_classes - 1).cuda() * self.push_scale

    def forward(self, x, labels, **kwargs):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        ncenters, nfeas = self.centers.size()
        distmat_x2cent = calc_distmat2(x, self.centers)
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        classes = Variable(classes)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dists_dcl = []
        dists_pull = []
        if not self.mode:
            _zero = torch.zeros(1).cuda()
            return _zero, _zero, _zero, _zero

        modes = self.mode.split('.')
        for i in range(batch_size):
            dist_pull = distmat_x2cent[i][mask[i]]
            dists_pull.append(dist_pull)
            if 'dcl' in modes:
                if 'min' in modes:
                    dist_push = distmat_x2cent[i][1 -
                                                  mask[i]].min() * self.push_wei
                else:
                    dist_push = (
                            distmat_x2cent[i][1 - mask[i]] * self.push_wei).mean()
                if 'with1' in modes:
                    dists_dcl.append(dist_pull / (dist_push + 1))
                else:
                    dists_dcl.append(dist_pull / dist_push)
            if 'margin' in modes:
                dists_dcl.append(
                    (torch.max(torch.zeros(1).cuda(),
                               dist_pull /
                               distmat_x2cent[i][1 - mask[i]] -
                               1 + self.margin2
                               ) * self.push_wei).mean()
                )
            if 'exp' in modes:
                # choose only most largest k value in negative
                logits = -distmat_x2cent[i]
                if self.args.topk != -1:
                    neg_topk, _ = torch.topk(logits[1 - mask[i]], k=self.args.topk, largest=True)
                    pos = logits[mask[i]]
                    logits = torch.cat([pos, neg_topk])

                shift_logits = logits - torch.max(logits)
                Z = torch.exp(shift_logits).sum()
                dist_now = shift_logits - torch.log(Z)

                # dist_now = F.log_softmax(logits, dim=0)
                dists_dcl.append(-dist_now[0])
                # dists_dcl
                # if 'nopos' in modes:
                #     dists_dcl.append(
                #         -torch.exp(-dist_pull) / torch.exp(-distmat_x2cent[i][1 - mask[i]]).sum()
                #     )
                # else:
                #     dists_dcl.append(
                #         - torch.exp(-dist_pull) / torch.exp(-distmat_x2cent[i]).sum()
                #     )

        loss_pull = torch.cat(dists_pull).mean()

        if 'cent' in modes:
            loss = loss_pull
        elif 'ccent' in modes:
            if dists_dcl[0].shape == ():
                dists_dcl = torch.stack(dists_dcl)
            else:
                dists_dcl = torch.cat(dists_dcl)
            loss = dists_dcl.mean()
        else:
            loss = torch.zeros(1).cuda()
        distmat_cent2cent = calc_distmat2(self.centers, self.centers)

        # if 'disall' in modes:
        mask = to_torch(np.tri(ncenters, dtype=np.uint8) -
                        np.identity(ncenters, dtype=np.uint8)).cuda()
        cent_pairs = distmat_cent2cent[mask]
        loss_dis = -cent_pairs.mean()
        # else:
        #     mask = to_torch(np.identity(ncenters, dtype=np.float32)).cuda() * distmat_cent2cent.max()
        #     loss_dis = (distmat_cent2cent + mask).min(dim=1)
        #     loss_dis = -loss_dis.mean()

        return loss, loss_dis, distmat_cent2cent, loss_pull


class QuadLoss(nn.Module):
    name = 'quad'

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
        # for numerical stability
        dist = dist.clamp(min=1e-12, max=1e+12).sqrt()
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

        dist_ap = _concat(dist_ap)
        dist_an = _concat(dist_an)
        dist_n12 = _concat(dist_n12)
        if torch.cuda.is_available():
            y = Variable(to_torch(np.ones(dist_an.size())).type(
                torch.cuda.FloatTensor), requires_grad=False).cuda()
        else:
            y = Variable(to_torch(np.ones(dist_an.size())).type(
                torch.FloatTensor), requires_grad=False)

        loss = self.ranking_loss(dist_an, dist_ap, y) + \
               0.1 * self.ranking_loss(dist_n12, dist_ap, y)
        # todo 0.1 and different margin
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)

        if not dbg:
            return loss, prec, dist
        else:
            return loss, prec, dist, dist_ap, dist_an


class QuinLoss(nn.Module):
    name = 'quin'

    def __init__(self, margin=0, mode='hard', **kwargs):
        super(QuinLoss, self).__init__()
        self.margin = margin
        self.mode = mode
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, dbg=False, cids=None):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        # for numerical stability
        dist = dist.clamp(min=1e-12, max=1e+12).sqrt()
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        view_mask = cids.expand(n, n).eq(cids.expand(n, n).t())
        dist_ap, dist_an, dist_n12 = [], [], []
        dist_p12 = []
        for i in range(n):
            some_pos = dist[i][mask[i]]
            some_neg = dist[i][mask[i] == 0]

            neg, n1_ind = some_neg.min(0)
            pos, p2_ind = some_pos.max(0)

            dist_ap.append(pos)
            dist_an.append(neg)

            n1_ind = to_numpy(n1_ind)[0]
            some_n2 = dist[n1_ind][mask[n1_ind] == 0]
            n2 = some_n2.min()
            dist_n12.append(n2)

            p2_ind = to_numpy(p2_ind)[0]
            # dist[i][mask[i]][view_mask[i][mask[i]]]
            some_p1 = dist[p2_ind][mask[p2_ind]]
            p1 = some_p1.max()
            dist_p12.append(p1)

        dist_ap = _concat(dist_ap)
        dist_an = _concat(dist_an)
        dist_n12 = _concat(dist_n12)
        dist_p12 = _concat(dist_p12)
        if torch.cuda.is_available():
            y = Variable(to_torch(np.ones(dist_an.size())).type(
                torch.cuda.FloatTensor), requires_grad=False).cuda()
        else:
            y = Variable(to_torch(np.ones(dist_an.size())).type(
                torch.FloatTensor), requires_grad=False)

        loss = self.ranking_loss(dist_an, dist_ap, y) \
               + 0.1 * self.ranking_loss(dist_n12, dist_ap, y) \
               + 0.1 * self.ranking_loss(dist_an, dist_p12, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)

        if not dbg:
            return loss, prec, dist
        else:
            return loss, prec, dist, dist_ap, dist_an


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1, targets.unsqueeze(1).data.cpu(), 1).cuda()
        targets = (1 - self.epsilon) * targets + \
                  self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    name = 'tri'

    def __init__(self, margin=0, mode='hard', args=None, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin != 'soft':
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mode = mode
        # self.margin2 = torch.tensor(args.margin2).cuda()
        # self.margin3 = torch.tensor(args.margin3).cuda()
        self.margin2 = args.margin2
        self.margin3 = args.margin3

    def forward(self, inputs, targets, dbg=False, cids=None):
        n = inputs.size(0)
        dist = calc_distmat(inputs, inputs)
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        # all_ind = to_variable(torch.arange(0, n).type(torch.LongTensor), requires_grad=False, volatile=True)
        # posp_inds, negp_inds = [], []

        for i in range(n):
            if self.mode == 'hard':
                some_pos = dist[i][mask[i]]
                some_neg = dist[i][mask[i] == 0]

                neg = some_neg.min()
                pos = some_pos.max()

                dist_ap.append(pos)
                dist_an.append(neg)
            elif self.mode == 'adap':
                some_pos = dist[i][mask[i]]
                some_neg = dist[i][mask[i] == 0]

                pos_ind = np.random.choice(np.arange(some_pos.shape[0]),
                                           p=F.softmax(some_pos, dim=0).cpu().detach().numpy())
                neg_ind = np.random.choice(np.arange(some_neg.shape[0]),
                                           p=F.softmax(-some_neg, dim=0).cpu().detach().numpy())

                neg = some_neg[neg_ind]
                pos = some_pos[pos_ind]

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
                dist_ap.append(posp[np.random.randint(0, posp.size(0))])

                negp = dist[i][mask[i] == 0]
                dist_an.append(negp[np.random.randint(0, negp.size(0))])

        dist_ap = _concat(dist_ap) * self.margin2
        dist_an = _concat(dist_an) / self.margin3
        y = torch.ones(dist_an.size(), requires_grad=False).cuda()
        if self.margin != 'soft':
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = F.softplus(dist_ap - dist_an).mean()
        prec = (dist_an.data > dist_ap.data).sum().type(
            torch.FloatTensor) / y.size(0)

        if not dbg:
            return loss, prec, dist
        else:
            return loss, prec, dist, dist_ap, dist_an


def _concat(l):
    l0 = l[0]
    if l0.shape == ():
        return torch.stack(l)
    else:
        return torch.cat(l)


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
