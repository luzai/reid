from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision

__all__ = ['Siamese', ]


def _make_conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1,
               bias=False):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=bias)
    init.kaiming_normal(conv.weight, mode='fan_out')
    init.constant(conv.bias, 0)
    init.constant(conv.weight, 1)

    bn = nn.BatchNorm2d(out_planes)
    init.constant(bn.bias, 0)

    relu = nn.ReLU(inplace=True)
    return nn.Sequential(conv, bn, relu)


def _make_fc(in_, out_, dp_=0):
    fc = nn.Linear(in_, out_)
    init.normal(fc.weight, std=0.001)
    init.constant(fc.bias, 0)

    dp = nn.Dropout(dp_)

    relu = nn.ReLU(inplace=True)
    return nn.Sequential(fc, dp, relu)


class Siamese(nn.Module):
    def __init__(self, dropout=0, mode='cat'):
        super(Siamese, self).__init__()
        self.dropout = dropout
        self.mode = mode

        self.conv1 = _make_conv(4096, 2048)
        self.avg1 = nn.AvgPool2d((4, 2))

        self.fc1 = _make_fc(2048, 1024, dp_=self.dropout)
        self.fc2 = _make_fc(1024, )

        self.reset_params()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)
