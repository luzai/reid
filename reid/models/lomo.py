from __future__ import absolute_import

from torch import nn
from lz import *
from torchvision.models.resnet import BasicBlock, model_zoo, model_urls
import torch, math
from reid.models.common import _make_conv, _make_fc
from reid.utils.serialization import load_state_dict


class LomoNet(nn.Module):
    def __init__(self, block=BasicBlock, layers= [2, 2, ], num_classes=1000):
        self.inplanes = 64
        super(LomoNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

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


if __name__ == '__main__':
    model = LomoNet(BasicBlock, [2, 2, ])
    x = Variable(torch.rand(1, 64, 40, 20))
    y = model(x)
    print(y.size())
    print(model)
