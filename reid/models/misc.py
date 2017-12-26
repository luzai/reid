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
        x = self.layer3(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        return x


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


class DoubleConv2(nn.Module):
    def __init__(self, in_plates=2048, out_plates=512, w=4, h=8, zmeta=4, z=3, stride=2):
        super(DoubleConv2, self).__init__()
        self.in_plates, self.out_plates, self.w, self.h, self.zmeta, self.z, self.stride = in_plates, out_plates, w, h, zmeta, z, stride
        self.weight = nn.Parameter(torch.FloatTensor(out_plates, in_plates, zmeta, zmeta))
        self.reset_parameters()

        self.n_inst, self.n_inst_sqrt = (self.zmeta - self.z + 1) * (self.zmeta - self.z + 1), self.zmeta - self.z + 1

    def reset_parameters(self):
        n = self.out_plates * self.zmeta * self.zmeta
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # n_inst_sqrt = self.n_inst_sqrt
        weight = []
        for i in range(self.zmeta - self.z + 1):
            for j in range(self.zmeta - self.z + 1):
                weight.append(self.weight[:, :, i:i + self.z, j:j + self.z])
        n_inst = len(weight)
        n_inst_sqrt = int(math.sqrt(n_inst))
        weight_inst = torch.cat(weight)
        # self.weight_inst = weight_inst

        bs = input.size(0)
        out = F.conv2d(input, weight_inst, padding=1)
        # self.out_inst = out
        out = out.view(bs, n_inst_sqrt, n_inst_sqrt, self.out_plates, self.h, self.w)
        out = out.permute(0, 3, 4, 5, 1, 2).contiguous().view(bs, -1, n_inst_sqrt, n_inst_sqrt)
        out = F.avg_pool2d(out, (self.stride, self.stride))
        out = out.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.h, self.w)
        # self.out=out
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(out.size(0), -1)
        return out


class DoubleConv3(nn.Module):
    def __init__(self, in_plates=2048, out_plates=512, w=4, h=8, zmeta=4, z=3, stride2=12, bs=100, controller=None):
        super(DoubleConv3, self).__init__()
        self.in_plates, self.out_plates, self.w, self.h, self.zmeta, self.z, self.stride2 = in_plates, out_plates, w, h, zmeta, z, stride2
        self.bs = bs
        self.weight = nn.Parameter(
            torch.FloatTensor(out_plates, in_plates, zmeta, zmeta))
        self.reset_parameters()
        self.register_buffer('theta', torch.FloatTensor(controller).view(-1, 2, 3).cuda())
        self.n_inst, self.n_inst_sqrt = (self.zmeta - self.z + 1) * (self.zmeta - self.z + 1), self.zmeta - self.z + 1

    def reset_parameters(self):
        n = self.out_plates * self.zmeta * self.zmeta
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        bs = input.size(0)

        weight_l = []
        for theta_ in Variable(self.theta, requires_grad=False):
            grid = F.affine_grid(theta_.expand(self.weight.size(0), 2, 3), self.weight.size())
            weight_l.append(F.grid_sample(self.weight, grid))
        weight_inst = torch.cat(weight_l)
        weight_inst = weight_inst[:, :, :3, :3]
        out = F.conv2d(input, weight_inst, padding=1)
        # self.out_inst = out
        # input.size(),weight_inst.size(),out.size()
        out = out.view(bs, self.stride2, self.out_plates, self.h, self.w)
        out = out.permute(0, 3, 4, 1, 2).contiguous().view(bs, -1, self.stride2)
        out = F.avg_pool1d(out, self.stride2)
        out = out.permute(0, 2, 1).contiguous().view(bs, -1, self.h, self.w)
        # self.out=out
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(out.size(0), -1)
        return out


if __name__ == '__main__':
    init_dev((3,))
    model = DoubleConv2(128, 64, stride=1).cuda()
    x = Variable(torch.rand(12, 128, 8, 4), requires_grad=False).cuda()
    y = model(x)
    y.sum().backward()

    print(model.weight.grad.size(), y.size())
