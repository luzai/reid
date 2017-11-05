from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class TupletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TupletLoss, self).__init__()
        self.margin = margin
        self.bce_with_logit = nn.BCEWithLogitsLoss()

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

        # Compute bce loss
        y = torch.from_numpy(np.concatenate((
            np.ones((n,)),
            np.zeros((n,))
        )))
        y = y.type_as(pairs)
        y.resize_as_(pairs)
        y = Variable(y)
        loss = self.bce_with_logit(inputs=pairs, targets=y)
        prec = ((pairs.data > 0.5) == y).sum() * 1. / y.size(0)
        return loss, prec
