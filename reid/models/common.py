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
    init.kaiming_normal_(conv.weight, mode='fan_out')
    if bias:
        init.constant(conv.bias, 0)

    bn = nn.BatchNorm2d(out_planes)
    # init.constant(bn.bias, 0)
    # init.constant(bn.weight, 1)

    if with_relu:
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)
    else:
        return nn.Sequential(conv, bn)


def _make_fc(in_, out_, dp_=0., with_relu=True, init_method='kaiming'):
    fc = nn.Linear(in_, out_)
    if init_method == 'normal':
        init.normal(fc.weight, std=0.001)
    else:
        init.kaiming_normal(fc.weight, mode='fan_out')
    init.constant(fc.bias, 0)
    dp = nn.Dropout(dp_)
    relu = nn.ReLU(inplace=True)
    res = [fc, ]
    if dp_ != 0:
        res.append(dp)
    if with_relu:
        res.append(relu)
    return nn.Sequential(*res)
