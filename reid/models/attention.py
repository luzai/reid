from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

__all__ = ['Attention', 'attention50']


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
    def __init__(self, in_planes, branch_dim, height, width, dopout=0.3, normalize=True):
        super(MaskBranch, self).__init__()
        self.height = height
        self.width = width
        # self.branch_left[0][1].bias
        self.branch_left = nn.Sequential(
            _make_conv(in_planes, 1, with_relu=False),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d((self.height // 32, self.width // 32))
        self.fc = _make_fc(in_planes, branch_dim, with_relu=not normalize, dp_=dopout)

    def forward(self, x):
        x_left = self.branch_left(x)
        # x[0,0,0,0], x_left[0,0,0,0] ,x_merge[0,0,0,0]
        x_merge = x * x_left
        x_merge = self.pool(x_merge)
        x_merge = x_merge.view(x_merge.size(0), -1)
        x_merge = self.fc(x_merge)
        return x_merge


class Mask(nn.Module):
    def __init__(self, in_planes, branchs, branch_dim, height, width, normalize=True, dopout=0.3, use_global=False):
        super(Mask, self).__init__()
        self.branch_l = nn.ModuleList()
        for ind in range(branchs):
            self.branch_l.append(MaskBranch(in_planes, branch_dim, height, width, dopout=dopout, normalize=normalize))
        if use_global:
            self.branch_g = [nn.AvgPool2d((height // 32, width // 32)),
                             _make_fc(in_planes, branch_dim, with_relu=not normalize, dp_=dopout)]
        self.normalize = normalize
        self.use_global = use_global

    def forward(self, x):

        x_l = torch.cat([b(x) for b in self.branch_l], 1)
        if self.use_global:
            x_g = self.branch_g[0](x)
            x_g= x_g.view(x_g.size(0), -1)
            x_g = self.branch_g[1](x_g)
            x = torch.cat([x_g, x_l], 1)
        else:
            x = x_l

        if self.normalize:
            x = F.normalize(x)

        return x


class Attention(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth=50, pretrained=True,
                 dropout=0.3, branchs=8,
                 branch_dim=64, height=256,
                 width=128, normalize=True, use_global=False, **kwargs):
        super(Attention, self).__init__()

        self.depth = depth
        self.pretrained = pretrained

        self.branchs = branchs
        self.branch_dim = branch_dim

        # Construct base (pretrained) resnet
        if depth not in Attention.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = Attention.__factory[depth](pretrained=pretrained)

        self.conv1 = _make_conv(2048, branchs * branch_dim, with_relu=not normalize)

        self.mask = Mask(branchs * branch_dim, branchs, branch_dim, height, width, dopout=dropout, normalize=normalize,
                         use_global=use_global)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        x = self.conv1(x)
        x = self.mask(x)

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


# def resnet18(**kwargs):
#     return ResNet(18, **kwargs)
#
#
# def resnet34(**kwargs):
#     return ResNet(34, **kwargs)


def attention50(**kwargs):
    return Attention(50, **kwargs)

#
# def resnet101(**kwargs):
#     return ResNet(101, **kwargs)
#
#
# def resnet152(**kwargs):
#     return ResNet(152, **kwargs)
