from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from reid.models.common import _make_fc, _make_conv


class MaskBranch(nn.Module):
    def __init__(self, in_planes, branch_dim, height, width, dopout=0.3):
        super(MaskBranch, self).__init__()
        self.height = height
        self.width = width
        # self.branch_left[0][1].bias
        self.branch_left = nn.Sequential(
            _make_conv(in_planes, 1, with_relu=False),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d((self.height, self.width))
        self.fc = _make_fc(in_planes, branch_dim, with_relu=False, dp_=dopout)

    def forward(self, x):
        x_left = self.branch_left(x)
        x_merge = x * x_left
        x_merge = self.pool(x_merge)
        x_merge = x_merge.view(x_merge.size(0), -1)
        x_merge = self.fc(x_merge)
        return x_merge


class Mask(nn.Module):
    def __init__(self, in_planes, branchs, branch_dim, height, width, dopout=0.3):
        super(Mask, self).__init__()
        self.branch_l = nn.ModuleList()
        for ind in range(branchs):
            self.branch_l.append(MaskBranch(in_planes, branch_dim, height, width, dopout=dopout))

    def forward(self, x):
        x = torch.cat([b(x) for b in self.branch_l], 1)
        return x


class Global(nn.Module):
    def __init__(self, in_planes, out_planes, dropout=0.3, ):
        super(Global, self).__init__()
        self.fc = _make_fc(in_planes, out_planes, dp_=dropout, with_relu=False)

    def forward(self, x):
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConcatReduce(nn.Module):
    def __init__(self, in_planes, out_planes, dropout=0, normalize=True, num_classes=None):
        super(ConcatReduce, self).__init__()
        self.normalize = normalize
        self.bn = nn.BatchNorm1d(in_planes)
        init.constant(self.bn.weight, 1)
        init.constant(self.bn.bias, 0)
        self.bn_relu = nn.ReLU()
        self.fc = _make_fc(in_planes, out_planes, dp_=dropout, with_relu=False)
        if num_classes is not None:
            self.fc2 = _make_fc(out_planes, num_classes, dp_=dropout    , with_relu=False )
        else:
            self.fc2 =None

    def forward(self, *input):
        if len(input) > 1:
            x = torch.cat(input, dim=1)
        else:
            x = input[0]
        x = self.bn(x)
        x = self.bn_relu(x)
        x = self.fc(x)
        if self.fc2 is None:
            return x
        else:
            return x, self.fc2(x)
