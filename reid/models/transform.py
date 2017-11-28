from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
import subprocess
from lz import *


class Transform(nn.Module):
    def __init__(self, mode='hard', **kwargs):
        super(Transform, self).__init__()
        self.mode = mode
        # subprocess.call(('rm -rf exps/dbg').split())
        # self.writer = SummaryWriter('./exps/dbg')
        # self.db = Database('./dbg.hard.h5','w')
        self.iter = 0

    def forward(self, inputs, targets, info=None, distmat =None ):
        n = inputs.size(0)
        all_ind = Variable(torch.arange(0, n).type(torch.LongTensor), requires_grad=False,volatile=True).cuda()

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        # # Compute pairwise distance, replace by the official when merged
        inputs_flat = inputs.view(inputs.size(0), -1)
        if distmat is None:
            dist = torch.pow(inputs_flat, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(1, -2, inputs_flat, inputs_flat.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        else:
            dist = distmat.cuda()

        pair1, pair2 = [], []
        pair2_ind = []

        # def get_hard_ind():

        if self.mode == 'hard':
            for i in range(n):
                pair1.append(inputs[i, :])
                posp_ind = all_ind[mask[i]]
                _, posp_ind_t = dist[i][mask[i]].max(0)
                posp_ind = posp_ind[posp_ind_t]
                pair2_ind.append(int(posp_ind.data.cpu().numpy()))
                posp = inputs[posp_ind, :]
                pair2.append(posp)
            for i in range(n):
                pair1.append(inputs[i, :])
                negp_ind = all_ind[mask[i] == 0]
                _, negp_ind_t = dist[i][mask[i] == 0].min(0)
                negp_ind = negp_ind[negp_ind_t]
                pair2_ind.append(int(negp_ind.data.cpu().numpy()))
                negp = inputs[negp_ind, :]
                pair2.append(negp)
        else:
            for i in range(n):
                pair1.append(inputs[i, :])
                posp_ind = all_ind[mask[i]]
                posp_ind_t = np.random.randint(0, posp_ind.size(0))
                posp_ind = posp_ind[posp_ind_t]
                posp = inputs[posp_ind, :]
                pair2_ind.append(int(posp_ind.data.cpu().numpy()))
                pair2.append(posp)
            for i in range(n):
                pair1.append(inputs[i, :])
                negp_ind = all_ind[mask[i] == 0]
                negp_ind_t = np.random.randint(0, negp_ind.size(0))
                negp_ind = negp_ind[negp_ind_t]
                pair2_ind.append(int(negp_ind.data.cpu().numpy()))
                negp = inputs[negp_ind, :]
                pair2.append(negp)

        pair1, pair2 = torch.stack(pair1), torch.cat(pair2)
        pair1.size(), pair2.size()
        y = torch.from_numpy(np.concatenate((
            np.ones((n,)),
            np.zeros((n,)),
            # np.ones((n,)),
            # np.zeros((n,))
        )))
        if info is not None:
            info['inds2'] = pair2_ind
        y = y.type_as(pair1.data)
        # y.resize_as_()
        y = Variable(y, requires_grad=False)

        # if self.iter % 10 == 0:
        #     self.db[self.iter] = np.asarray(pair2_ind)
        #     self.writer.add_histogram('features', inputs, self.iter)
        #     self.writer.add_histogram('dist', dist, self.iter)
        #     self.writer.add_histogram('ap', dist_ap, self.iter)
        #     self.writer.add_histogram('an', dist_an, self.iter)

        self.iter += 1
        # y.size()
        return pair1, pair2, y, info
