import h5py
import torch

from collections import OrderedDict


def cvt_bn(f, in_name, state_dict, out_name):
    state_dict[out_name + '.weight'] = torch.from_numpy(
        f[in_name + '.s@value'][...])
    state_dict[out_name + '.bias'] = torch.from_numpy(f[in_name
                                                        + '.b@value'][...])
    dim = len(f[in_name + '.b@value'])
    state_dict[out_name + '.running_mean'] = torch.from_numpy(
        f[in_name + '.h@value'][:dim])
    state_dict[out_name + '.running_var'] = torch.from_numpy(
        f[in_name + '.h@value'][dim:])


def cvt_conv_fc(f, in_name, state_dict, out_name):
    state_dict[out_name + '.weight'] = torch.from_numpy(
        f[in_name + '.w@value'][...])
    if in_name + '.b@value' in f:
        state_dict[out_name + '.bias'] = torch.from_numpy(
            f[in_name + '.b@value'][...])


def cvt_fc(f, in_name, state_dict, out_name):
    state_dict[out_name + '.weight'] = torch.from_numpy(
        f[in_name + '.w@value'][...].squeeze())
    if in_name + '.b@value' in f:
        state_dict[out_name + '.bias'] = torch.from_numpy(
            f[in_name + '.b@value'][...].squeeze())


def adjust_input_layer(state_dict, name, rgb_std=[0.229, 0.224, 0.225]):
    w = state_dict[name + '.weight']
    w = w[:, torch.LongTensor([2, 1, 0]), :, :] * 255
    w *= torch.Tensor(rgb_std)[:, None, None]
    state_dict[name + '.weight'] = w


def main(parrots_model, out_file, adjust_input=False):
    layers = [3, 4, 23, 3]
    block_depth = 3
    state_dict = OrderedDict()
    f = h5py.File(parrots_model, 'r')
    cvt_conv_fc(f, 'conv1', state_dict, 'conv1')
    if adjust_input:
        adjust_input_layer(state_dict, 'conv1')
    cvt_bn(f, 'bn1', state_dict, 'bn1')
    for i in range(len(layers)):
        for k in range(1, block_depth + 1):
            cvt_conv_fc(f, 'res{}a.conv{}'.format(i + 2, k), state_dict,
                        'layer{}.0.conv{}'.format(i + 1, k))
            cvt_bn(f, 'res{}a.bn{}'.format(i + 2, k), state_dict,
                   'layer{}.0.bn{}'.format(i + 1, k))
        if 'res{}a.shortcut.w@value'.format(i + 2) in f:
            cvt_conv_fc(f, 'res{}a.shortcut'.format(i + 2), state_dict,
                        'layer{}.0.downsample.0'.format(i + 1))
            cvt_bn(f, 'res{}a.shortcut_bn'.format(i + 2), state_dict,
                   'layer{}.0.downsample.1'.format(i + 1))
        for j in range(1, layers[i]):
            for k in range(1, block_depth + 1):
                cvt_conv_fc(f, 'res{}b{}.conv{}'.format(i + 2, j, k),
                            state_dict, 'layer{}.{}.conv{}'.format(
                                i + 1, j, k))
                cvt_bn(f, 'res{}b{}.bn{}'.format(i + 2, j, k), state_dict,
                       'layer{}.{}.bn{}'.format(i + 1, j, k))
    cvt_conv_fc(f, 'rpn_output', state_dict, 'rpn_top.rpn_conv')
    cvt_conv_fc(f, 'rpn_cls_score', state_dict, 'rpn_top.rpn_cls_score')
    cvt_conv_fc(f, 'rpn_bbox_pred', state_dict, 'rpn_top.rpn_bbox_pred')
    cvt_fc(f, 'cls_score', state_dict, 'fc_cls')
    cvt_fc(f, 'bbox_pred', state_dict, 'fc_reg')
    f.close()
    torch.save({'state_dict': state_dict}, out_file)


if __name__ == '__main__':
    main(
        '/mnt/gv7/wjq/faster_rcnn_190/resnet101_v1_multi_scale_viddet50/snapshots/iter.latest.parrots',
        'frcnn_resnet101.pth')
