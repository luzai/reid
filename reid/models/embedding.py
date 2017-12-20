import math
from torch import nn

from .kron import KronMatching
from .common import _make_conv,_make_fc
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

class ConcatEmbed(nn.Module):
    def __init__(self, in_planes):
        super(ConcatEmbed, self).__init__()
        self.conv1 = _make_conv(in_planes, 1024)
        self.conv2 = _make_conv(1024, 1024)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = _make_fc(1024, 512, dp_=0.4)
        self.fc2 = _make_fc(512, 2, dp_=0.3,with_relu=False)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        # self.conv1[1].eval()
        # self.conv2[1].eval()
        # self.fc1[1].eval()
        # self.fc2[1].eval()

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class EltwiseSubEmbed(nn.Module):
    def __init__(self, nonlinearity='square', use_batch_norm=False,
                 use_classifier=False, num_features=0, num_classes=0):
        super(EltwiseSubEmbed, self).__init__()
        self.nonlinearity = nonlinearity
        if nonlinearity is not None and nonlinearity not in ['square', 'abs']:
            raise KeyError("Unknown nonlinearity:", nonlinearity)
        self.use_batch_norm = use_batch_norm
        self.use_classifier = use_classifier
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(num_features)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        if self.use_classifier:
            assert num_features > 0 and num_classes > 0
            self.classifier = nn.Linear(num_features, num_classes)
            self.classifier.weight.data.normal_(0, 0.001)
            self.classifier.bias.data.zero_()

    def forward(self, x1, x2):
        x = x1 - x2
        if self.nonlinearity == 'square':
            x = x.pow(2)
        elif self.nonlinearity == 'abs':
            x = x.abs()
        if self.use_batch_norm:
            x = self.bn(x)
        if self.use_classifier:
            x = self.classifier(x)
        else:
            x = x.sum(1)
            x = x.clamp(min=1e-12).sqrt()
        return x


class KronEmbed(nn.Module):
    def __init__(self, height, width, out_channels, num_classes):
        super(KronEmbed, self).__init__()
        self.kron = KronMatching()
        in_channels = (2 * height - 1) * (2 * width - 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.pool = nn.AvgPool2d((height, width))
        self.fc = nn.Linear(out_channels, num_classes)

        self.reset_params()

    def forward(self, x1, x2):
        x = self.kron(x1, x2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
