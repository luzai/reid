from __future__ import absolute_import

from torch.nn import init
from torchvision.models.resnet import model_zoo, model_urls
from reid.utils.serialization import load_state_dict
from .common import _make_conv, _make_fc

from lz import *

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


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


from reid.models.misc import DoubleConv4,DoubleConv5

ConvOp = DoubleConv4
# ConvOp = DoubleConv5

class ResNetOri(nn.Module):

    def __init__(self, block, layers, num_classes=1000, **kwargs):
        self.inplanes = 64
        super(ResNetOri, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], dc=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dc=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dc=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dc=True)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.out_planes = self.fc.in_features
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dc=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)
                # if not dc else
                # ConvOp(self.inplanes, planes * block.expansion,
                #        kernel_size=1, stride=stride, bias=False)
                ,
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dc))
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dc=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if not dc:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        else:
            self.conv2 = ConvOp(planes, planes, kernel_size=3, stride=stride,
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


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetOri(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        load_state_dict(model, model_zoo.load_url(model_urls['resnet50']))
    return model


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, **kwargs):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # self.stn = STN_TPS()
        # self.stn = STN_shallow()

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)
        out_planes = self.base.fc.in_features
        self.out_planes = out_planes
        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
                self.feat_relu = nn.ReLU()
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)
        # self.conv1 = _make_conv(out_planes, 512, kernel_size=1, stride=1, padding=0, with_relu=True)
        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        # x, loss = self.stn(x)

        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)
        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
            x = self.feat_relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
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


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


# def resnet50(**kwargs):
#     return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
