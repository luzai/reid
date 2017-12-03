from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch


def _make_conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1,
               bias=False, with_relu=True):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=bias)
    init.kaiming_normal(conv.weight, mode='fan_out')
    if bias:
        init.constant(conv.bias, 0)

    bn = nn.BatchNorm2d(out_planes)
    init.constant(bn.bias, 0)
    init.constant(bn.weight, 1)

    if with_relu:
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)
    else:
        return nn.Sequential(conv, bn)


def _make_fc(in_, out_, dp_=0., with_relu=True):
    fc = nn.Linear(in_, out_)
    init.normal(fc.weight, std=0.001)
    init.constant(fc.bias, 0)
    dp = nn.Dropout(dp_)
    relu = nn.ReLU(inplace=True)
    res = [fc, ]
    if dp_ != 0:
        res.append(dp)
    if with_relu:
        res.append(relu)
    return nn.Sequential(*res)


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
    def __init__(self, in_planes, out_planes, dropout=0.3):
        super(Global, self).__init__()
        self.fc = _make_fc(in_planes, out_planes, dp_=dropout, with_relu=False)

    def forward(self, x):
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConcatReduce(nn.Module):
    def __init__(self, in_planes, out_planes, dropout=0, normalize=True):
        super(ConcatReduce, self).__init__()
        self.normalize = normalize
        self.bn = nn.BatchNorm1d(in_planes)
        self.fc = _make_fc(in_planes, out_planes, dp_=dropout, with_relu=False)

    def forward(self, *input):
        if len(input) > 1:
            x = torch.cat(input, dim=1)
        else:
            x = input[0]
        x = self.bn(x)
        x = F.normalize(x)
        x = self.fc(x)

        return x
