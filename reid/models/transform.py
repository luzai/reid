from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
import subprocess


class Transform(nn.Module):
    def __init__(self, mode='hard', **kwargs):
        super(Transform, self).__init__()
        self.mode = mode
        # subprocess.call(('rm -rf exps/dbg').split())
        # self.writer = SummaryWriter('./exps/dbg')
        # self.iter = 0

    def forward(self, inputs, targets):
        n = inputs.size(0)
        all_ind = Variable(torch.arange(0, n).type(torch.LongTensor), requires_grad=False).cuda()

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        # # Compute pairwise distance, replace by the official when merged
        inputs_flat = inputs.view(inputs.size(0), -1)
        dist = torch.pow(inputs_flat, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs_flat, inputs_flat.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        pair1, pair2 = [], []
        # if self.mode == 'rand'
        for i in range(n):
            pair1.append(inputs[i, :])
            posp_ind = all_ind[mask[i]]
            # _, posp_ind_t = dist[i][mask[i]].max(0)
            # posp_ind = posp_ind[Variable(posp_ind_t.data, requires_grad=False)]
            posp_ind = np.random.randint(0, posp_ind.size(0))
            # posp = torch.index_select(inputs, 0, posp_ind)
            posp = inputs[posp_ind:posp_ind + 1, :]
            pair2.append(posp)
        # pair1[0].size(),pair2[0].size()
        for i in range(n):
            pair1.append(inputs[i, :])
            negp_ind = all_ind[mask[i] == 0]
            # _, negp_ind_t = dist[i][mask[i] == 0].min(0)
            # negp_ind = negp_ind[Variable(negp_ind_t.data, requires_grad=False)]
            negp_ind = np.random.randint(0, negp_ind.size(0))
            # negp = torch.index_select(inputs, 0, negp_ind)
            negp = inputs[negp_ind:negp_ind + 1, :]
            pair2.append(negp)

        pair1, pair2 = torch.stack(pair1), torch.cat(pair2)
        pair1.size(), pair2.size()
        y = torch.from_numpy(np.concatenate((
            np.ones((n,)),
            np.zeros((n,))
        )))

        y = y.type_as(pair1.data)
        # y.resize_as_()
        y = Variable(y, requires_grad=False)

        # if self.iter % 10 == 0:
        #     self.writer.add_histogram('features', inputs, self.iter)
        #     self.writer.add_histogram('dist', dist, self.iter)
        #     self.writer.add_histogram('ap', dist_ap, self.iter)
        #     self.writer.add_histogram('an', dist_an, self.iter)
        #
        # self.iter += 1
        # y.size()
        return pair1, pair2, y
