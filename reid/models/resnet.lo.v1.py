from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def _make_conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1,
               bias=False, with_relu=True):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=bias)
    init.kaiming_normal(conv.weight, mode='fan_out')
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
    def __init__(self, in_planes, branch_dim, dopout=0.3):
        super(MaskBranch, self).__init__()

        self.branch_left = nn.Sequential(
            _make_conv(in_planes, 1, kernel_size=1, stride=1, padding=0 , with_relu=False),
            nn.Sigmoid()
        )

        self.fc = _make_fc(in_planes, branch_dim, with_relu=False, dp_=dopout)

    def forward(self, x):
        x_left = self.branch_left(x)
        # x[0,0,0,0], x_left[0,0,0,0] ,x_merge[0,0,0,0]
        x_merge = x * x_left
        x_merge = F.avg_pool2d(x_merge, x_merge.size()[2:])
        x_merge = x_merge.view((x_merge.size(0), -1))

        x_merge = self.fc(x_merge)
        return x_merge


class Mask(nn.Module):
    def __init__(self, in_planes, branchs, branch_dim, dopout=0.3):
        super(Mask, self).__init__()
        self.branch_l = nn.ModuleList()
        for ind in range(branchs):
            self.branch_l.append(MaskBranch(in_planes, branch_dim, dopout=dopout))

    def forward(self, x):
        x = torch.cat([b(x) for b in self.branch_l], 1)

        return x


def stat(t):
    return t.min(), t.max(), t.size()


# stat(ori)

class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        # depth = 50, pretrained =True, cut_at_pooling=False, num_features=1024, norm =False,dropout=0,num_classes=128
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)
            self.ori_bn = nn.BatchNorm2d(2048)
            self.mask = Mask(512, 8, 64, dopout=dropout)
            self.conv1 = _make_conv(2048, 512, kernel_size=1, stride=1, padding=0,  with_relu=False)
        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x
        x = self.conv1(x)

        x = self.mask(x)

        # if self.has_embedding:
        #     x = self.feat(x)
        #     x = self.feat_bn(x)
        # if self.norm:
        # x = F.relu(x)
        x = F.normalize(x)
        # elif self.has_embedding:

        # x = torch.cat([x, x2], dim=1)
        # if self.dropout > 0:
        # x = self.drop(x)
        # if self.num_classes > 0:
        #     x = self.classifier(x)
        return x

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


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
