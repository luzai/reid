from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
from torch.nn import init
from reid.utils.serialization import *


class MobileNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MobileNet, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.feat = nn.Linear(1024, num_features)
        self.feat_bn = nn.BatchNorm1d(self.num_features)
        init.kaiming_normal(self.feat.weight, mode='fan_out')
        init.constant(self.feat.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.classifier.weight, std=0.001)
        init.constant(self.classifier.bias, 0)
        state_dict = load_checkpoint('/home/xinglu/.torch/models/mobilenet.pth')['state_dict']
        load_state_dict(self.model, state_dict, own_de_prefix='module.model.')

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.feat(x)
        x = self.feat_bn(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    str = "0123 is string example....wow!!!0000000"
    str.lstrip('0123')
    str.lstrip('3210')
    model = MobileNet(1024, 128)
