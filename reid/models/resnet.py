from torch.nn import init
from torchvision.models.resnet import conv3x3, model_urls, model_zoo
from lz import *
from reid.utils.serialization import load_state_dict
from .common import _make_conv

import sys

sys.path.insert(0, '/data1/xinglu/prj/deformable-convolution-pytorch/')

from modules import ConvOffset2d
from functions import conv_offset2d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(downsample[0], downsample[1],
                          kernel_size=1, stride=downsample[2], bias=False),
                nn.BatchNorm2d(downsample[1]),
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(downsample[0], downsample[1],
                          kernel_size=1, stride=downsample[2], bias=False),
                nn.BatchNorm2d(downsample[1]),
            )
        else:
            self.downsample = None
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


def reset_params(module, zero=False):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            if zero:
                m.weight.data.fill_(0)
            else:
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


class Unet(nn.Module):
    def __init__(self, inchannels, outchannels, rep=1):
        super(Unet, self).__init__()
        unet_block = UnetBlock(inchannels, outchannels, innermost=True)
        for i in range(rep - 1):
            unet_block = UnetBlock(inchannels, outchannels, submodule=unet_block)
        self.model = unet_block

    def forward(self, x):
        return self.model(x)


class UnetBlock(nn.Module):
    def __init__(self, inchannels, outchannels,
                 submodule=None, innermost=False
                 ):
        super(UnetBlock, self).__init__()
        down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        up = nn.UpsamplingBilinear2d(scale_factor=2)

        if innermost:
            conv = nn.Sequential(
                conv3x3(inchannels, outchannels),
                nn.BatchNorm2d(outchannels),
                nn.ReLU()
            )
            model = [down, conv, up]
        else:
            downconv = nn.Sequential(
                conv3x3(inchannels, inchannels),
                nn.BatchNorm2d(inchannels),
                nn.ReLU()
            )
            upconv = nn.Sequential(
                conv3x3(outchannels, outchannels),
                nn.BatchNorm2d(outchannels),
                nn.ReLU()
            )
            model = [down, downconv, submodule, upconv, up]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + self.model(x)


# deform conv
class DeformConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=False,
                 num_deformable_groups=1,
                 unet_rep=0):
        super(DeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.unet_rep = unet_rep
        if unet_rep == 0:
            self.conv_offset = nn.Conv2d(in_channels,
                                         num_deformable_groups * 2 * kernel_size * kernel_size,
                                         kernel_size=kernel_size,
                                         stride=stride, padding=padding, bias=bias)
        else:
            self.conv_offset = Unet(in_channels,
                                    num_deformable_groups * 2 * kernel_size * kernel_size, rep=unet_rep)
        self.conv = ConvOffset2d(in_channels, out_channels, stride=stride, padding=padding,
                                 kernel_size=kernel_size,
                                 num_deformable_groups=num_deformable_groups)
        self.weight = self.conv.weight
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        reset_params(self.conv_offset, True)
        if self.bias:
            self.conv_offset.bias.data.zero_()
        self.conv.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        offset = self.conv_offset(x)
        output = self.conv(x, offset)
        return output


def get_norm(x):
    x2 = x * x
    x2 = x2.sum()
    x2 = to_numpy(x2)
    x2 = np.sqrt(x2)
    return x2


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, mult=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        self.mult = mult

    #     self.reset_params()
    #
    # def reset_params(self):
    #     pass

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # logging.info('se: ori is {}, now is {}'.format(get_norm(x), get_norm(x * y)))
        return x * y * self.mult


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, **kwargs):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(downsample[0], downsample[1],
                          kernel_size=1, stride=downsample[2], bias=False),
                nn.BatchNorm2d(downsample[1]),
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, **kwargs):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(downsample[0], downsample[1],
                          kernel_size=1, stride=downsample[2], bias=False),
                nn.BatchNorm2d(downsample[1]),
            )
        else:
            self.downsample = None
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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class XSEXFXBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, **kwargs):
        super(XSEXFXBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction, mult=1)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(downsample[0], downsample[1],
                          kernel_size=1, stride=downsample[2], bias=False),
                nn.BatchNorm2d(downsample[1]),
            )
        else:
            self.downsample = None
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

        # logging.info('after1x1conv: residual is {}, identity is {}'.format(get_norm(out), get_norm(residual)))

        out += (residual + self.se(residual))
        out = self.relu(out)

        return out


class AttResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, **kwargs):
        super(AttResBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction, mult=2)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(downsample[0], downsample[1],
                          kernel_size=1, stride=downsample[2], bias=False),
                nn.BatchNorm2d(downsample[1]),
            )
        else:
            self.downsample = None
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

        # logging.info('after1x1conv: residual is {}, identity is {}'.format(get_norm(out), get_norm(residual)))

        out += self.se(residual)
        out = self.relu(out)

        return out


class DeformBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(DeformBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DeformConv(planes,
                                planes,
                                kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(downsample[0], downsample[1],
                          kernel_size=1, stride=downsample[2], bias=False),
                nn.BatchNorm2d(downsample[1]),
            )
        else:
            self.downsample = None
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


class UnetDeformBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(UnetDeformBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DeformConv(planes,
                                planes,
                                kernel_size=3, stride=stride,
                                padding=1, bias=False, unet_rep=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(downsample[0], downsample[1],
                          kernel_size=1, stride=downsample[2], bias=False),
                nn.BatchNorm2d(downsample[1]),
            )
        else:
            self.downsample = None
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


class SEDeformBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, **kwargs):
        super(SEDeformBottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.conv2 = DeformConv(planes,
                                planes,
                                kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        # self.se = SELayer(planes * 4, reduction)
        self.relu = nn.ReLU(inplace=True)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(downsample[0], downsample[1],
                          kernel_size=1, stride=downsample[2], bias=False),
                nn.BatchNorm2d(downsample[1]),
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.se(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # out = self.se(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RIRBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(RIRBottleneck, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = []
        self.bn = []
        for i in range(4):
            self.conv += [nn.Conv2d(inplanes // 2, planes // 2, kernel_size=1, bias=False), ]
            self.bn += [nn.BatchNorm2d(planes // 2)]

        for i in range(4, 8):
            self.conv += [conv3x3(planes // 2, planes // 2, stride)]
            self.bn += [nn.BatchNorm2d(planes // 2)]

        for i in range(8, 10):
            self.conv += [nn.Conv2d(planes // 2, planes * 2, kernel_size=1, bias=False)]
            self.bn += [nn.BatchNorm2d(planes * 2)]

        for i in range(10):
            setattr(self, 'conv' + str(i), self.conv[i])
            setattr(self, 'bn' + str(i), self.bn[i])

        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(downsample[0] // 2, downsample[1] // 2,
                          kernel_size=1, stride=downsample[2], bias=False),
                nn.BatchNorm2d(downsample[1] // 2),
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        residual, transient = [x[:, :x.size(1) // 2, :, :].contiguous(), x[:, x.size(1) // 2:, :, :].contiguous()]
        init_residual = residual
        func = lambda ind, x: self.bn[ind](self.conv[ind](x))
        out = [func(ind, residual) for ind in range(2)] + \
              [func(ind, transient) for ind in range(2, 4)]

        residual, transient = (
            # residual +
            out[0] + out[2],
            out[1] + out[3])
        residual, transient = map(self.relu, (residual, transient))

        func = lambda ind, x: self.bn[ind](self.conv[ind](x))
        out = [func(ind, residual) for ind in range(4, 6)] + \
              [func(ind, transient) for ind in range(6, 8)]
        residual, transient = (
            # residual +
            out[0] + out[2],
            out[1] + out[3])
        residual, transient = map(self.relu, (residual, transient))

        func = lambda ind, x: self.bn[ind](self.conv[ind](x))
        out = [func(8, residual), func(9, transient)]
        if self.downsample:
            init_residual = self.downsample(init_residual)
        residual, transient = (init_residual + out[0]), out[1]
        residual, transient = map(self.relu, (residual, transient))
        return torch.cat((residual, transient), dim=1).contiguous()


class RIRBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(RIRBasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = []
        self.bn = []
        for i in range(4):
            self.conv += [conv3x3(inplanes // 2, planes // 2, stride)]
            self.bn += [nn.BatchNorm2d(planes // 2)]

        for i in range(4, 8):
            self.conv += [conv3x3(planes // 2, planes // 2)]
            self.bn += [nn.BatchNorm2d(planes // 2)]
        for i in range(8):
            setattr(self, 'conv' + str(i), self.conv[i])
            setattr(self, 'bn' + str(i), self.bn[i])

        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(downsample[0] // 2, downsample[1] // 2,
                          kernel_size=1, stride=downsample[2], bias=False),
                nn.BatchNorm2d(downsample[1] // 2),
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        residual, transient = x[:, :x.size(1) // 2, :, :].contiguous(), x[:, x.size(1) // 2:, :, :].contiguous()
        func = lambda ind, x: self.bn[ind](self.conv[ind](x))
        out = [func(ind, residual) for ind in range(2)] + \
              [func(ind, transient) for ind in range(2, 4)]
        if self.downsample:
            residual = self.downsample(residual)

        residual, transient = (residual + out[0] + out[2],
                               out[1] + out[3])
        residual, transient = map(self.relu, (residual, transient))

        func = lambda ind, x: self.bn[ind](self.conv[ind](x))
        out = [func(ind, residual) for ind in range(4, 6)] + \
              [func(ind, transient) for ind in range(6, 8)]

        residual, transient = (residual + out[0] + out[2],
                               out[1] + out[3])
        residual, transient = map(self.relu, (residual, transient))
        return torch.cat((residual, transient), dim=1).contiguous()


class SERIRBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SERIRBasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = []
        self.bn = []
        for i in range(4):
            self.conv += [conv3x3(inplanes // 2, planes // 2, stride)]
            self.bn += [nn.BatchNorm2d(planes // 2)]

        for i in range(4, 8):
            self.conv += [conv3x3(planes // 2, planes // 2)]
            self.bn += [nn.BatchNorm2d(planes // 2)]
        self.se1 = SELayer(planes // 2)
        self.se2 = SELayer(planes // 2)
        for i in range(8):
            setattr(self, 'conv' + str(i), self.conv[i])
            setattr(self, 'bn' + str(i), self.bn[i])

        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(downsample[0] // 2, downsample[1] // 2,
                          kernel_size=1, stride=downsample[2], bias=False),
                nn.BatchNorm2d(downsample[1] // 2),
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        residual, transient = x[:, :x.size(1) // 2, :, :].contiguous(), x[:, x.size(1) // 2:, :, :].contiguous()
        func = lambda ind, x: self.bn[ind](self.conv[ind](x))
        out = [func(ind, residual) for ind in range(2)] + \
              [func(ind, transient) for ind in range(2, 4)]
        if self.downsample:
            residual = self.downsample(residual)

        residual, transient = (residual + out[0] + out[2],
                               out[1] + out[3])
        residual, transient = map(self.relu, (residual, transient))

        func = lambda ind, x: self.bn[ind](self.conv[ind](x))
        out = [func(ind, residual) for ind in range(4, 6)] + \
              [func(ind, transient) for ind in range(6, 8)]
        out[2] = self.se1(out[2])
        out[3] = self.se2(out[3])
        residual, transient = (residual + out[0] + out[2],
                               out[1] + out[3])
        residual, transient = map(self.relu, (residual, transient))
        return torch.cat((residual, transient), dim=1).contiguous()


class ResNet(nn.Module):
    __factory = {
        '18': [2, 2, 2, 2],
        '34': [3, 4, 6, 3],
        '50': [3, 4, 6, 3],
        '101': [3, 4, 23, 3],
        '152': [3, 8, 36, 3],
    }

    def _make_layer(self, block, planes, blocks, stride=1):
        if not isinstance(block, list):
            block = [block] * blocks
        downsample = None
        if stride != 1 or self.inplanes != planes * block[0].expansion:
            downsample = (self.inplanes, planes * block[0].expansion, stride)
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes * block[0].expansion,
            #               kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(planes * block[0].expansion),
            # )

        layers = []
        layers.append(block[0](self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block[0].expansion
        for i in range(1, blocks):
            layers.append(block[i](self.inplanes, planes))

        return nn.Sequential(*layers)

    def __init__(self, depth=50, pretrained=True,
                 cut_at_pooling=False,
                 num_features=0, dropout=0,
                 num_classes=0, block_name='Bottleneck',
                 block_name2='Bottleneck',
                 num_deform=3, **kwargs):
        super(ResNet, self).__init__()
        depth = str(depth)
        self.depth = depth
        self.inplanes = 64
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        self.dropout = dropout
        self.num_classes = num_classes
        self.num_features = num_features

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        layers = ResNet.__factory[depth]
        block = eval(block_name)
        block2 = eval(block_name2)
        self.out_planes = 512 * block2.expansion
        logging.info(f'out_planes is {self.out_planes}')
        self.conv1 = nn.Conv2d(3, 64,
                               kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        if num_deform > 3:
            self.layer3 = self._make_layer([block] * 3 + [block2] * (num_deform - 3), 256, layers[2],
                                           stride=2)
            self.layer4 = self._make_layer([block2] * 3, 512, layers[3], stride=2)
        else:
            self.layer3 = self._make_layer([block] * 6, 256, layers[2], stride=2)
            self.layer4 = self._make_layer([block] * (3 - num_deform) + [block2] * num_deform, 512,
                                           layers[3], stride=2)
        self.post1 = nn.Sequential(
            nn.BatchNorm2d(self.out_planes),
            nn.ReLU(),
            nn.Dropout2d(self.dropout),
            nn.AdaptiveAvgPool2d(1),
        )
        self.post2 = nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(self.out_planes, self.num_features, bias=False),
        )
        self.post3 = nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(self.num_features, self.num_classes, bias=False),
        )

        reset_params(self.post1)
        reset_params(self.post2)
        reset_params(self.post3)

        if pretrained:
            # if 'SE' in block_name or 'SE' in block_name2:
            #     logging.info('load senet')
            #     state_dict = torch.load('/data1/xinglu/prj/pytorch-classification/work/se_res/model_best.pth.tar')[
            #         'state_dict']
            #     load_state_dict(self, state_dict, own_de_prefix='module.')
            # else:
            logging.info('load resnet')
            load_state_dict(self, model_zoo.load_url(model_urls['resnet{}'.format(depth)]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x1 = self.post1(x)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.post2(x1)

        x2 = self.post3(
            # Variable(x1.data),
            x1
        )

        return x1, x2


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


'''
def get_loss_div(theta):
    norm = theta.norm(p=2, dim=1, keepdim=True)
    theta = theta / norm
    mat = torch.matmul(theta, theta.transpose(1, 0))
    batch_size = mat.size(0)
    loss = (mat.sum() - mat.diag().sum()) / (batch_size * (batch_size - 1))
    return loss


class STN_res18(nn.Module):
    def __init__(self, ):
        super(STN_res18, self).__init__()
        self.loc = torchvision.models.resnet18(pretrained=True)

        self.fc = nn.Linear(self.loc.fc.in_features, 6)
        # init.constant(self.fc.weight, 0)
        self.fc.bias.data = self.fc.bias.data + torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, input):
        xs = input
        for name, module in self.loc._modules.items():
            if name == 'avgpool':
                break
            xs = module(xs)
        xs = F.avg_pool2d(xs, xs.size()[2:])
        xs = xs.view(xs.size(0), -1)
        theta = self.fc(xs)
        loss_div = get_loss_div(theta)

        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, input.size())
        x = F.grid_sample(input, grid)
        # loss_sim = (x-input).mean()
        loss = loss_div
        return x, loss


class STN_shallow(nn.Module):
    def __init__(self, ):
        super(STN_shallow, self).__init__()
        self.loc = nn.Sequential(
            _make_conv(3, 64, kernel_size=7, stride=8, padding=0),
            _make_conv(64, 128, kernel_size=7, stride=4, padding=2),
        )
        self.fc = nn.Linear(128, 6)
        # init.constant(self.fc.weight, 0)
        self.fc.bias.data = self.fc.bias.data + torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, input):
        xs = input
        xs = self.loc(xs)
        xs = F.avg_pool2d(xs, xs.size()[2:])
        xs = xs.view(xs.size(0), -1)
        theta = self.fc(xs)
        loss_div = get_loss_div(theta)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, input.size())
        x = F.grid_sample(input, grid)
        loss = 5e-2 * loss_div
        return x, loss


from tps_grid_gen import TPSGridGen
from grid_sample import grid_sample


class UnBoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(UnBoundedGridLocNet, self).__init__()
        self.conv = nn.Sequential(
            _make_conv(3, 64, kernel_size=7, stride=4, padding=2),
            _make_conv(64, 128, kernel_size=7, stride=4, padding=2),
            _make_conv(128, 256, kernel_size=3, stride=2, padding=1),
        )
        self.fc = nn.Linear(256, grid_height * grid_width * 2)

        bias = target_control_points.view(-1)
        self.fc.bias.data.copy_(bias)
        # self.fc.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        points = self.fc(x)
        return points.view(batch_size, -1, 2)


class STN_TPS(nn.Module):
    def __init__(self):
        super(STN_TPS, self).__init__()
        r1 = r2 = 0.9
        self.grid_height = grid_height = 16
        self.grid_width = grid_width = 8
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (grid_width - 1)),
        )))
        Y, X = target_control_points.split(1, dim=1)
        target_control_points = torch.cat([X, Y], dim=1)

        self.loc_net = UnBoundedGridLocNet(grid_height, grid_width, target_control_points)
        self.tps = TPSGridGen(256, 128, target_control_points)

    def forward(self, x):
        batch_size = x.size(0)
        source_control_points = self.loc_net(x)
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(batch_size, 256, 128, 2)
        transformed_x = grid_sample(x, grid)
        return transformed_x, 0


from reid.models.misc import *


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dc=False, convop='nn.Conv2d'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if not dc:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        else:
            self.conv2 = eval(convop)(planes, planes, kernel_size=3, stride=stride,
                                      padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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


class Bottleneck2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dc=False, convop='nn.Conv2d'):
        super(Bottleneck2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if not dc:
            self.conv2 = nn.Conv2d(planes, planes * 4, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        else:
            self.conv2 = eval(convop)(planes, planes * 4, kernel_size=3, stride=stride,
                                      padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# bn sum relu
class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock2, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = []
        self.bn = []
        for i in range(4):
            self.conv += [conv3x3(inplanes // 2, planes // 2, stride)]
            self.bn += [nn.BatchNorm2d(planes // 2)]

        for i in range(4, 8):
            self.conv += [conv3x3(planes // 2, planes // 2)]
            self.bn += [nn.BatchNorm2d(planes // 2)]
        for i in range(8):
            setattr(self, 'conv' + str(i), self.conv[i])
            setattr(self, 'bn' + str(i), self.bn[i])

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual, transient = x

        func = lambda ind, x: self.relu(self.bn[ind](self.conv[ind](x)))
        out = [func(ind, residual) for ind in range(2)] + \
              [func(ind, transient) for ind in range(2, 4)]
        if self.downsample:
            residual = self.downsample(residual)

        residual, transient = (residual + out[0] + out[2],
                               out[1] + out[3])

        func = lambda ind, x: self.bn[ind](self.conv[ind](x))
        out = [func(ind, residual) for ind in range(4, 6)] + [func(ind, transient) for ind in range(6, 8)]

        residual, transient = (residual + out[0] + out[2],
                               out[1] + out[3])

        return (residual, transient)


# sum bn relu
class BasicBlock3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = []
        self.bn = []
        for i in range(4):
            self.conv += [conv3x3(inplanes // 2, planes // 2, stride)]
        self.bn += [nn.BatchNorm2d(planes // 2)]

        for i in range(4, 8):
            self.conv += [conv3x3(planes // 2, planes // 2)]
        self.bn += [nn.BatchNorm2d(planes // 2)]

        for i in range(8):
            setattr(self, 'conv' + str(i), self.conv[i])
        for i in range(2):
            setattr(self, 'bn' + str(i), self.bn[i])

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual, transient = x
        func = lambda ind, x: self.conv[ind](x)
        out = [func(ind, residual) for ind in range(2)] + \
              [func(ind, transient) for ind in range(2, 4)]
        if self.downsample:
            residual = self.downsample(residual)

        residual, transient = (residual + out[0] + out[2],
                               out[1] + out[3])
        residual, transient = map(lambda x: self.relu(self.bn[0](x)), (residual, transient))

        func = lambda ind, x: self.conv[ind](x)
        out = [func(ind, residual) for ind in range(4, 6)] + \
              [func(ind, transient) for ind in range(6, 8)]

        residual, transient = (residual + out[0] + out[2],
                               out[1] + out[3])
        residual, transient = map(lambda x: self.relu(self.bn[1](x)), (residual, transient))

        return (residual, transient)


class ResNet34(nn.Module):

    def __init__(self, block, layers, num_classes=1000, **kwargs):
        self.inplanes = 64
        self.out_planes = 512
        self.bottleneck = kwargs.get('bottleneck', 'BasicBlock')
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes if self.bottleneck != 'BasicBlock2' else self.inplanes // 2,
                          planes * block.expansion if self.bottleneck != 'BasicBlock2' else planes * block.expansion // 2,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(
                    planes * block.expansion if self.bottleneck != 'BasicBlock2' else planes * block.expansion // 2),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.bottleneck == 'BasicBlock2':
            x = [x[:, :x.size(1) // 2, :, :], x[:, x.size(1) // 2:, :, :]]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.bottleneck == 'BasicBlock2':
            x = torch.cat(x, dim=1).contiguous()
        return x


class ResNet50(nn.Module):

    def __init__(self, block, layers, convop='nn.Conv2d', num_classes=1000, **kwargs):
        self.inplanes = 64
        self.out_planes = 2048
        super(ResNet50, self).__init__()
        self.convop = convop
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], dc=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dc=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dc=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dc=True)

    def _make_layer(self, block, planes, blocks, stride=1, dc=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)
                # if not dc else
                # eval(self.convop)(self.inplanes, planes * block.expansion,
                #        kernel_size=1, stride=stride, bias=False)
                ,
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            inplanes=self.inplanes,
            planes=planes, stride=stride, downsample=downsample, dc=dc, convop=self.convop))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.

       Args:
           pretrained (bool): If True, returns a model pre-trained on ImageNet
       """
    bottleneck = kwargs.get('bottleneck')

    model = ResNet34(eval(bottleneck), [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        load_state_dict(model, model_zoo.load_url(model_urls['resnet34']))
    return model

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, **kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.out_planes = 2048
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    bottleneck = kwargs.get('bottleneck')
    convop = kwargs.get('convop')
    if bottleneck == 'SEBottleneck':
        model = ResNet(eval(bottleneck), [3, 4, 6, 3], **kwargs)
    else:
        model = ResNet50(eval(bottleneck), [3, 4, 6, 3], **kwargs)
    if pretrained:
        if bottleneck == 'SEBottleneck':
            state_dict = torch.load('/data1/xinglu/prj/senet.pytorch/weight-99.pkl')['weight']
            state_dict.keys()
            load_state_dict(model, state_dict, own_de_prefix='module.')
        else:
            # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
            load_state_dict(model, model_zoo.load_url(model_urls['resnet50']))
    return model



class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(Residual, self).__init__()
        self.downsample = downsample

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1,
                               stride=1 if not downsample else 2
                               )
        self.bn3 = nn.BatchNorm2d(out_channels // 4)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, 1)
        if downsample:
            self.conv4 = nn.Conv2d(in_channels, out_channels, 1,
                                   stride=2)
        elif in_channels != out_channels:
            self.conv4 = nn.Conv2d(in_channels, out_channels, 1, stride=1)
        else:
            self.conv4 = None

    def forward(self, x):

        residual = x
        x = F.relu(self.bn1(x))
        x = self.conv1(x)
        x = F.relu(self.bn2(x))
        x = self.conv2(x)
        x = F.relu(self.bn3(x))
        x = self.conv3(x)

        if self.conv4 is not None:
            residual = self.conv4(residual)

        return x + residual


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, unet_rep=1):
        super(Attention, self).__init__()
        self.residual1 = Residual(in_channels, in_channels)
        self.unet = nn.Sequential(Unet(in_channels, rep=unet_rep),
                                  nn.Conv2d(in_channels, in_channels, 1),
                                  nn.Conv2d(in_channels, in_channels, 1),
                                  nn.Sigmoid()
                                  )
        self.trunk = nn.Sequential(Residual(in_channels, in_channels),
                                   Residual(in_channels, in_channels))
        self.residual2 = Residual(in_channels, in_channels)

    def forward(self, x):
        x = self.residual1(x)
        x1 = self.trunk(x)
        x2 = self.unet(x)
        x3 = x1 * x2 + x1
        return self.residual2(x3)


class ResAtt1(nn.Module):
    def __init__(self, **kwargs):
        super(ResAtt1, self).__init__()
        self.out_planes = 2048
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual1 = Residual(64, 256)
        self.attention1 = Attention(256, 256, unet_rep=3)
        self.residual2 = Residual(256, 512, downsample=True)
        self.attention2 = Attention(512, 512, unet_rep=2)
        self.residual3 = Residual(512, 1024, downsample=True)
        self.attention3 = Attention(1024, 1024, unet_rep=1)
        self.residual4 = nn.Sequential(Residual(1024, 2048, downsample=True),
                                       Residual(2048, 2048, ),
                                       Residual(2048, 2048, ),
                                       )
        # self.fc1 = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.residual1(x)
        x = self.attention1(x)
        x = self.residual2(x)
        x = self.attention2(x)
        x = self.residual3(x)
        x = self.attention3(x)
        x = self.residual4(x)
        # x = F.avg_pool2d(x, x.size()[-2:])
        # x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        return x


class Unet(nn.Module):
    def __init__(self, nc, rep=1):
        super(Unet, self).__init__()
        unet_block = UnetBlock(nc, innermost=True)
        for i in range(rep - 1):
            unet_block = UnetBlock(nc, submodule=unet_block)
        self.model = unet_block

    def forward(self, x):
        return self.model(x)


class UnetBlock(nn.Module):
    def __init__(self, nc,
                 submodule=None, innermost=False
                 ):
        super(UnetBlock, self).__init__()
        downconv = Residual(nc, nc)
        upconv = Residual(nc, nc)
        down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        up = nn.UpsamplingBilinear2d(scale_factor=2)

        if innermost:
            model = [down, downconv, upconv, up]
        else:
            model = [down, downconv, submodule, upconv, up]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + self.model(x)


def res_att1(**kwargs):
    return ResAtt1(**kwargs)

'''
