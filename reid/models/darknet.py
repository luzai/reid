from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
from torch.nn import init

class DarkNet(nn.Module):
    def __init__(self,num_features,num_classes):
        self.num_features = num_features
        self.num_classes=num_classes
        super(DarkNet, self).__init__()
        self.feat_model = nn.Sequential(OrderedDict([
            # conv1
            ('conv1', nn.Conv2d(3, 16, 3, 1, 1, bias=False)),
            ('bn1', nn.BatchNorm2d(16)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(2, 2)),
            # conv2
            ('conv2', nn.Conv2d(16, 32, 3, 1, 1, bias=False)),
            ('bn2', nn.BatchNorm2d(32)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(2, 2)),
            # conv3
            ('conv3', nn.Conv2d(32, 64, 3, 1, 1, bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU()),
            ('pool3', nn.MaxPool2d(2, 2)),
            # conv4
            ('conv4', nn.Conv2d(64, 128, 3, 1, 1, bias=False)),
            ('bn4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU()),
            ('pool4', nn.MaxPool2d(2, 2)),
            # conv5
            ('conv5', nn.Conv2d(128, 256, 3, 1, 1, bias=False)),
            ('bn5', nn.BatchNorm2d(256)),
            ('relu5', nn.ReLU()),
            ('pool5', nn.MaxPool2d(2, 2)),
            # conv6
            ('conv6', nn.Conv2d(256, 512, 3, 1, 1, bias=False)),
            ('bn6', nn.BatchNorm2d(512)),
            ('relu6', nn.ReLU()),
            ('pool6', nn.MaxPool2d(2, 2)),
            # conv7
            ('conv7', nn.Conv2d(512, 1024, 3, 1, 1, bias=False)),
            ('bn7', nn.BatchNorm2d(1024)),
            ('relu7', nn.ReLU())
        ]))
        self.load_weights('/home/xinglu/.torch/models/tiny-yolo.weights')
        self.feat =nn.Linear(1024,num_features)
        self.feat_bn = nn.BatchNorm1d(self.num_features)
        init.kaiming_normal(self.feat.weight, mode='fan_out')
        init.constant(self.feat.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.classifier.weight, std=0.001)
        init.constant(self.classifier.bias, 0)

    def load_weights(self, path):
        buf = np.fromfile(path, dtype=np.float32)
        start = 4
        start = load_conv_bn(buf, start,
                             self.feat_model[0], self.feat_model[1])
        start = load_conv_bn(buf, start,
                             self.feat_model[4], self.feat_model[5])
        start = load_conv_bn(buf, start,
                             self.feat_model[8], self.feat_model[9])
        start = load_conv_bn(buf, start,
                             self.feat_model[12], self.feat_model[13])
        start = load_conv_bn(buf, start,
                             self.feat_model[16], self.feat_model[17])
        start = load_conv_bn(buf, start,
                             self.feat_model[20], self.feat_model[21])

    def forward(self, x):
        x = self.feat_model(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x=self.feat(x)
        x=self.feat_bn(x)
        x=self.classifier(x)
        return x


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    # print(num_w,num_b)
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]))
    start = start + num_w
    return start


if __name__ == '__main__':
    model = DarkNet()
    model.load_weights('/home/xinglu/.torch/models/tiny-yolo.weights')

