import torch.nn as nn
import torch
from easydict import EasyDict as edict

cfg = edict()
cfg.use_limbs = False


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * 2,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * 2),
        )
        
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


from lz import F


class refineNet(nn.Module):
    def __init__(self, lateral_channel, out_shape, num_class, args=None):
        super(refineNet, self).__init__()
        cascade = []
        num_cascade = 4
        for i in range(num_cascade):
            cascade.append(self._make_layer(lateral_channel, num_cascade - i - 1, out_shape))
        self.cascade = nn.ModuleList(cascade)
        self.final_predict = self._predict(4 * lateral_channel, num_class)
        self.args = args
        if cfg.use_limbs:
            self.final_predict2 = self._predict(4 * lateral_channel, cfg.nlimbs * 2)
    
    def _make_layer(self, input_channel, num, output_shape):
        layers = []
        for i in range(num):
            layers.append(Bottleneck(input_channel, 128))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        return nn.Sequential(*layers)
    
    def _predict(self, input_channel, num_class):
        # layers = []
        # layers.append(Bottleneck(input_channel, 128))
        # layers.append(nn.Conv2d(256, num_class,
        #     kernel_size=3, stride=1, padding=1, bias=False))
        # layers.append(nn.BatchNorm2d(num_class))
        # return nn.Sequential(*layers)
        return nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(input_channel, num_class, ),
            nn.BatchNorm1d(num_class),
            nn.ReLU(),
        )
    
    def forward(self, x, ft=False):
        if ft:
            torch.set_grad_enabled(False)
        bs = x[0].size(0)
        refine_fms = []
        for i in range(4):
            refine_fms.append(self.cascade[i](x[i]))
        out_fea = torch.cat(refine_fms, dim=1)
        out_fea = F.adaptive_avg_pool2d(out_fea, 1).view(bs, -1)
        if ft:
            torch.set_grad_enabled(True)
        out = self.final_predict(out_fea)
        if not cfg.use_limbs:
            return out  # todo fx
        else:
            return out, self.final_predict2(out_fea)
