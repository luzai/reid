import torch.nn as nn
import torch
import math
from lz import F


class globalNet(nn.Module):
    def __init__(self, channel_settings, output_shape, num_class):
        super(globalNet, self).__init__()
        self.channel_settings = channel_settings
        laterals, upsamples, predict = [], [], []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
            predict.append(self._predict(output_shape, num_class))
            if i != len(channel_settings) - 1:
                upsamples.append(self._upsample())
        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)
        self.predict = nn.ModuleList(predict)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _lateral(self, input_size):
        layers = []
        layers.append(nn.Conv2d(input_size, 256,
                                kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _upsample(self):
        layers = []
        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(256, 256,
                                      kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        
        return nn.Sequential(*layers)
    
    def _predict(self, output_shape, num_class):
        # layers = []
        # layers.append(nn.Conv2d(256, 256,
        #     kernel_size=1, stride=1, bias=False))
        # layers.append(nn.BatchNorm2d(256))
        # layers.append(nn.ReLU(inplace=True))
        #
        # layers.append(nn.Conv2d(256, num_class,
        #     kernel_size=3, stride=1, padding=1, bias=False))
        # layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        # layers.append(nn.BatchNorm2d(num_class))
        #
        # return nn.Sequential(*layers)
        return nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(256, num_class, ),
            nn.BatchNorm1d(num_class),
            nn.ReLU(),
        )
    
    def forward(self, x, ft=False):
        bs = x[0].size(0)
        global_fms, global_outs = [], []
        for i in range(len(self.channel_settings)):
            if ft:
                torch.set_grad_enabled(False)
            if i == 0:
                feature = self.laterals[i](x[i])
            else:
                feature = self.laterals[i](x[i]) + up
            global_fms.append(feature)
            if i != len(self.channel_settings) - 1:
                up = self.upsamples[i](feature)
            feature = F.adaptive_avg_pool2d(feature, 1).view(bs, -1)
            if ft:
                torch.set_grad_enabled(True)
            feature = self.predict[i](feature)
            global_outs.append(feature)
        return global_fms, global_outs
