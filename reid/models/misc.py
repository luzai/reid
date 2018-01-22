from __future__ import absolute_import

from torchvision.models.resnet import BasicBlock, model_zoo, model_urls

from lz import *
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
        x = self.layer3(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        return x


# original doubly conv
class DoubleConv(nn.Module):
    def __init__(self, cl=2048, cl2=512, w=8, h=4, zp=4, z=3, s=2):
        super(DoubleConv, self).__init__()
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


# deform conv
'''
from modules import ConvOffset2d
from functions import conv_offset2d

# deform conv
class DeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                 compression_ratio=1, num_deformable_groups=2):
        super(DeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv_offset = nn.Conv2d(in_channels, num_deformable_groups * 2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv = ConvOffset2d(in_channels, out_channels, stride=stride, padding=padding,
                                 kernel_size=kernel_size,
                                 num_deformable_groups=num_deformable_groups)
        self.weight = self.conv.weight

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.srqt(n)
        self.conv_offset.weight.data.uniform_(-stdv, stdv)
        self.conv.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        offset = self.conv_offset(x)
        output = self.conv(x, offset)
        return output


# deform+stn
class DeformConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=False, meta_kernel_size=None, compression_ratio=1, num_deformable_groups=2):
        super(DeformConv2, self).__init__()
        if meta_kernel_size is None:
            meta_kernel_size = kernel_size * 2

        def get_controller(
                scale=(1,
                       ),
                translation=(0,
                             2 / (meta_kernel_size - 1)
                             ),
                theta=(0,
                        # np.pi,
                        # np.pi / 8, -np.pi / 8,
                        # np.pi / 2, -np.pi / 2,
                        # np.pi / 4, -np.pi / 4,
                        # np.pi * 3 / 4, -np.pi * 3 / 4,
                       )
        ):
            controller = []
            for sx in scale:
                for sy in scale:
                    for tx in translation:
                        for ty in translation:
                            for th in theta:
                                controller.append([sx * np.cos(th), -sx * np.sin(th), tx,
                                                   sy * np.sin(th), sy * np.cos(th), ty])
            print('controller stride is ', len(controller))
            controller = np.stack(controller)
            controller = controller.reshape(-1, 2, 3)
            controller = np.ascontiguousarray(controller, np.float32)
            return controller

        while out_channels % compression_ratio != 0:
            compression_ratio += 1
            if compression_ratio >= out_channels:
                compression_ratio = 1
                break
        controller = get_controller()
        len_controller = len(controller)
        assert out_channels % compression_ratio == 0
        self.meta_kernel_size = meta_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_deformable_groups = num_deformable_groups
        self.weight = nn.Parameter(torch.FloatTensor(
            out_channels // compression_ratio, in_channels, meta_kernel_size, meta_kernel_size
        ))
        self.conv_offset_weight = nn.Parameter(torch.FloatTensor(
            num_deformable_groups * 2 * self.kernel_size * self.kernel_size, in_channels, kernel_size, kernel_size
        ))
        self.reset_parameters()
        self.register_buffer('theta', torch.FloatTensor(controller).view(-1, 2, 3).cuda())
        # self.reduce_conv = nn.Conv2d(out_channels * len_controller // compression_ratio, out_channels, 1)

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.conv_offset_weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        bs, chl, h, w = input.size()

        offset = F.conv2d(input, self.conv_offset_weight, stride=self.stride, padding=self.padding)
        weight_l = []
        for theta_ in Variable(self.theta, requires_grad=False):
            grid = F.affine_grid(theta_.expand(self.weight.size(0), 2, 3), self.weight.size())
            weight_l.append(F.grid_sample(self.weight, grid))
        weight_inst = torch.cat(weight_l)
        weight_inst = weight_inst[:, :, :self.kernel_size, :self.kernel_size].contiguous()
        out = conv_offset2d(input, offset, weight_inst, stride=self.stride, padding=self.padding,
                            deform_groups=self.num_deformable_groups)
        bs, _, oh, ow = out.size()
        # out = self.reduce_conv(out)
        out = out.view(bs, len(weight_l), self.out_channels, oh, ow)
        out = out.permute(0, 2, 3, 4, 1).contiguous().view(bs, -1, len(weight_l))
        out = F.avg_pool1d(out, len(weight_l))
        out = out.permute(0, 2, 1).contiguous().view(bs, -1, oh, ow)

        return out
'''

from orn.modules import ORConv2d
from orn.functions import oralign1d


class ORNConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=False, meta_kernel_size=None, compression_ratio=1):
        super(ORNConv, self).__init__()
        self.meta_kernel_size = meta_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        assert (kernel_size == 3 or kernel_size == 1, 'kernel size')
        assert (in_channels % 8 == 0 and out_channels % 8 == 0)
        self.conv = ORConv2d(in_channels // 8, out_channels // 8, kernel_size=kernel_size,
                             arf_config=(8, 8),
                             stride=stride, padding=padding, bias=bias
                             )
        self.weight = self.conv.weight

    def forward(self, input):
        return self.conv(input)

# my designed 2048 concat 2048
class GroupConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, meta_kernel_size=4,
                 compression_ratio=1):
        super(GroupConv, self).__init__()
        while out_channels % compression_ratio != 0:
            compression_ratio += 1
            if compression_ratio >= out_channels:
                compression_ratio = 1
                break
        assert out_channels % compression_ratio == 0
        self.meta_kernel_size = meta_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(
            out_channels // compression_ratio, in_channels, meta_kernel_size, meta_kernel_size
        ))
        self.inter_weight = nn.Parameter(torch.randn(
            out_channels // compression_ratio * 4, out_channels // compression_ratio * 4, 1, 1
        ))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        bs, chl, h, w = input.size()
        weight_l = []
        for i in range(self.meta_kernel_size - self.kernel_size + 1):
            for j in range(self.meta_kernel_size - self.kernel_size + 1):
                weight_l.append(self.weight[:, :, i:i + self.kernel_size, j:j + self.kernel_size])
        weight_inst1 = torch.cat(weight_l, dim=0).contiguous()
        weight_inst2 = F.conv2d(weight_inst1.permute(1, 0, 2, 3).contiguous(), self.inter_weight).permute(1, 0, 2,
                                                                                                          3).contiguous()
        weight_inst = torch.cat((weight_inst1, weight_inst2), dim=0).contiguous()
        out = F.conv2d(input, weight_inst, stride=self.stride, padding=self.padding)
        bs, _, oh, ow = out.size()
        # out = self.reduce_conv(out)
        out = out.view(bs, self.out_channels, len(weight_l) * 2, oh, ow)
        out = out.permute(0, 1, 3, 4, 2).contiguous().view(bs, -1, 2 * len(weight_l))
        out = F.avg_pool1d(out, len(weight_l) * 2)
        out = out.contiguous().view(bs, -1, oh, ow)

        return out


global_compression_ratio = 1

# move stack + reduce conv/avg pool
class TC1Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False, meta_kernel_size=4,
                 compression_ratio=global_compression_ratio, mode='train'):
        super(TC1Conv, self).__init__()
        while out_channels % compression_ratio != 0:
            compression_ratio += 1
            if compression_ratio >= out_channels:
                compression_ratio = 1
                break
        assert out_channels % compression_ratio == 0
        self.meta_kernel_size = kernel_size + 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mode = mode

        self.weight = nn.Parameter(torch.randn(
            out_channels // compression_ratio, in_channels, meta_kernel_size, meta_kernel_size
        ))
        len_controller = (self.meta_kernel_size - self.kernel_size + 1) ** 2
        self.reduce_conv = nn.Conv2d(out_channels * len_controller // compression_ratio,
                                     out_channels, kernel_size=1)

        self.reset_parameters()

    def get_weight_inst(self):
        weight_l = []

        for i in range(self.meta_kernel_size - self.kernel_size + 1):
            for j in range(self.meta_kernel_size - self.kernel_size + 1):
                weight_l.append(self.weight[:, :, i:i + self.kernel_size, j:j + self.kernel_size])
        weight_inst = torch.cat(weight_l).contiguous()
        return weight_inst

    def reset_parameters(self):
        def reset_w(weight):
            out_chl, in_chl, w, h = weight.size()
            n = in_chl * w * h
            stdv = 1. / math.sqrt(n)
            weight.data.uniform_(-stdv, stdv)

        reset_w(self.weight)
        reset_w(self.reduce_conv.weight)

    def forward(self, input):
        bs, chl, h, w = input.size()
        weight_inst = self.get_weight_inst()
        out = F.conv2d(input, weight_inst, stride=self.stride, padding=self.padding)
        bs, _, oh, ow = out.size()
        out = self.reduce_conv(out)
        # out = out.view(bs, len(weight_l), self.out_channels, oh, ow)
        # out = out.permute(0, 2, 3, 4, 1).contiguous().view(bs, -1, len(weight_l))
        # out = F.avg_pool1d(out, len(weight_l))
        # out = out.permute(0, 2, 1).contiguous().view(bs, -1, oh, ow)

        return out


# 1x1 conv generate weight
class C1Conv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=False, meta_kernel_size=4,
                 compression_ratio=global_compression_ratio, mode='train'):
        super(C1Conv, self).__init__()
        while out_channels % compression_ratio != 0:
            compression_ratio += 1
            if compression_ratio >= out_channels:
                compression_ratio = 1
                break
        assert out_channels % compression_ratio == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.meta_kernel_size = kernel_size + 1
        self.mode = mode
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.FloatTensor(
            out_channels // compression_ratio, in_channels, kernel_size, kernel_size
        ))
        self.trans_weight = nn.Parameter(torch.FloatTensor(
            out_channels, out_channels, 1, 1
        ))
        self.reset_parameters()

    def reset_parameters(self):
        def reset_w(weight):
            out_chl, in_chl, w, h = weight.size()
            n = in_chl * w * h
            stdv = 1. / math.sqrt(n)
            weight.data.uniform_(-stdv, stdv)

        reset_w(self.weight)
        reset_w(self.trans_weight)

    def get_weight_inst(self):
        weight_inst = F.conv2d(
            self.weight.permute(1, 0, 2, 3).contiguous(),
            self.trans_weight
        ).permute(1, 0, 2, 3).contiguous()
        return weight_inst

    def forward(self, input):
        bs, chl, h, w = input.size()
        if self.mode == 'test' and hasattr(self, 'weight_inst'):
            weight_inst = self.weight_inst
        elif self.mode == 'test' and not hasattr(self, 'weight_inst'):
            weight_inst = self.get_weight_inst()
            self.weight_inst = weight_inst
        else:
            weight_inst = self.get_weight_inst()
        out = F.conv2d(input, weight_inst, stride=self.stride, padding=self.padding)
        bs, _, oh, ow = out.size()
        return out


# double 1x1 conv generate weight
class C1C1Conv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=False, meta_kernel_size=4,
                 compression_ratio=global_compression_ratio, mode='train'):
        super(C1C1Conv, self).__init__()
        while out_channels % compression_ratio != 0:
            compression_ratio += 1
            if compression_ratio >= out_channels:
                compression_ratio = 1
                break
        assert out_channels % compression_ratio == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.meta_kernel_size = kernel_size + 1
        self.mode = mode
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.FloatTensor(
            out_channels // compression_ratio, in_channels, kernel_size, kernel_size
        ))
        self.trans_weight = nn.Parameter(torch.FloatTensor(
            out_channels, out_channels, 1, 1
        ))
        self.trans_weight2 = nn.Parameter(torch.FloatTensor(
            out_channels, out_channels, 1, 1
        ))
        self.reset_parameters()

    def reset_parameters(self):
        def reset_w(weight):
            out_chl, in_chl, w, h = weight.size()
            n = in_chl * w * h
            stdv = 1. / math.sqrt(n)
            weight.data.uniform_(-stdv, stdv)

        reset_w(self.weight)
        reset_w(self.trans_weight)
        reset_w(self.trans_weight2)

    def get_weight_inst(self):
        weight_inst = F.conv2d(
            self.weight.permute(1, 0, 2, 3).contiguous(),
            self.trans_weight
        ).permute(1, 0, 2, 3).contiguous()
        weight_inst = F.conv2d(
            weight_inst.permute(1, 0, 2, 3).contiguous(),
            self.trans_weight2
        ).permute(1, 0, 2, 3).contiguous()
        return weight_inst

    def forward(self, input):
        bs, chl, h, w = input.size()
        # if self.mode == 'test' and hasattr(self, 'weight_inst'):
        #     weight_inst = self.weight_inst
        # elif self.mode == 'test' and not hasattr(self, 'weight_inst'):
        #     weight_inst = self.get_weight_inst()
        #     self.weight_inst = weight_inst
        # else:
        weight_inst = self.get_weight_inst()

        out = F.conv2d(input, weight_inst, stride=self.stride, padding=self.padding)

        bs, _, oh, ow = out.size()
        return out


# zeropad + 1x1 conv generate weight
class ZPC1Conv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=False, meta_kernel_size=4,
                 compression_ratio=global_compression_ratio, mode='train'):
        super(ZPC1Conv, self).__init__()
        while out_channels % compression_ratio != 0:
            compression_ratio += 1
            if compression_ratio >= out_channels:
                compression_ratio = 1
                break
        assert out_channels % compression_ratio == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.meta_kernel_size = kernel_size + 1
        self.mode = mode
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(
            out_channels // compression_ratio, in_channels, kernel_size, kernel_size
        ))
        self.trans_weight = nn.Parameter(torch.randn(
            out_channels,
            out_channels // compression_ratio * len(range(0, 3, conf.get('meta_stride', 2))) ** 2,
            1, 1
        ))
        self.reset_parameters()

    def get_param(self):
        return cnt_param(self.weight),cnt_param(self.trans_weight)

    def reset_parameters(self):
        def reset_w(weight):
            out_chl, in_chl, w, h = weight.size()
            n = in_chl * w * h
            stdv = 1. / math.sqrt(n)
            weight.data.uniform_(-stdv, stdv)

        reset_w(self.weight)
        reset_w(self.trans_weight)

    def get_weight_inst(self):
        weight_meta = F.pad(self.weight, (1, 1, 1, 1), mode='constant', value=0)
        weight_l = []
        for i in range(0, 3, conf.get('meta_stride', 2)):
            for j in range(0, 3, conf.get('meta_stride', 2)):
                weight_l.append(weight_meta[:, :, i:i + 3, j:j + 3])
        weight_inst = torch.cat(weight_l).contiguous()
        weight_inst = F.conv2d(
            weight_inst.permute(1, 0, 2, 3).contiguous(),
            self.trans_weight
        ).permute(1, 0, 2, 3).contiguous()
        return weight_inst

    def forward(self, input):
        bs, chl, h, w = input.size()
        weight_inst = self.get_weight_inst()
        out = F.conv2d(input, weight_inst, stride=self.stride, padding=self.padding)
        bs, _, oh, ow = out.size()
        return out
