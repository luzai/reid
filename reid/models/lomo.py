from __future__ import absolute_import

from lz import *
from torchvision.models.resnet import BasicBlock, model_zoo, model_urls
import torch, math
from reid.models.common import _make_conv, _make_fc
from reid.utils.serialization import load_state_dict


class LomoNet(nn.Module):  # Bottleneck 2222 or 3463
    def __init__(self, block=BasicBlock, layers=[2, 2, 2], num_classes=1000):
        self.inplanes = 64
        self.out_planes = 128
        super(LomoNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        load_state_dict(self, model_zoo.load_url(model_urls['resnet18']))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        return x


class DConv(nn.Module):
    def __init__(self, cl=2048, cl2=1024, w=8, h=4, zp=4, z=3, s=2):
        super(DConv, self).__init__()
        self.cl, self.cl2, self.w, self.h, self.zp, self.z, self.s = cl, cl2, w, h, zp, z, s
        self.Wl = nn.Parameter(torch.Tensor(cl2, cl, zp, zp))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.cl
        for k in (self.zp, self.zp):
            n *= k
        stdv = 1. / math.sqrt(n)
        self.Wl.data.uniform_(-stdv, stdv)

    def forward(self, input):
        cl, cl2, w, h, zp, z, s = self.cl, self.cl2, self.w, self.h, self.zp, self.z, self.s

        Il = torch.eye(cl * z * z)
        Il = Il.view(cl * z * z, cl, z, z)
        Il = to_variable(Il, requires_grad=False)
        Wtl = F.conv2d(self.Wl, Il)
        zpz = zp - z + 1
        Wtl = Wtl.view(cl2 * zpz * zpz, cl, z, z)
        Ol2 = F.conv2d(input, Wtl)
        bs, _, wl2, hl2 = Ol2.size()
        Ol2 = Ol2.view(bs, -1, zpz, zpz)
        Il2 = F.avg_pool2d(Ol2, (s, s))
        res = Il2.view(bs, -1, wl2, hl2)
        res = F.avg_pool2d(res, res.size()[2:])
        res = res.view(res.size(0), -1)
        return res


if __name__ == '__main__':
    init_dev((3))
    model = DConv(128, 64).cuda()
    x = Variable(torch.rand(12, 128, 8, 4), requires_grad=False).cuda()
    y = model(x)
    y.sum().backward()
    x.grad
    model.Wl.grad
    print(y.size())
