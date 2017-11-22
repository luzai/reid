import h5py
import numpy as np
import torch
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls

def convert_bn(th_dict, th_layer, fout, pt_layer):
    fout[pt_layer + '.s@value'] = th_dict[th_layer + '.weight'].cpu().data.numpy()
    fout[pt_layer + '.b@value'] = th_dict[th_layer + '.bias'].cpu().data.numpy()
    fout[pt_layer + '.h@value'] = np.concatenate([
        th_dict[th_layer + '.running_mean'].cpu().numpy(),
        th_dict[th_layer + '.running_var'].cpu().numpy()
    ])


def convert_conv(th_dict, th_layer, fout, pt_layer):
    fout[pt_layer + '.w@value'] = th_dict[th_layer + '.weight'].cpu().data.numpy()
    if th_layer + '.bias' in th_dict:
        fout[pt_layer + '.b@value'] = th_dict[th_layer + '.bias'].cpu().data.numpy()


def convert_fc(th_dict, th_layer, fout, pt_layer):
    fout[pt_layer + '.w@value'] = th_dict[th_layer + '.weight'].cpu().data.numpy()
    fout[pt_layer + '.b@value'] = th_dict[th_layer + '.bias'].cpu().data.numpy()

def adjust_input_layer(state_dict, name, rgb_std=[0.229, 0.224, 0.225]):
    w = state_dict[name + '.weight']
    w = w[:, torch.LongTensor([2, 1, 0]), :, :] * 255
    w *= torch.Tensor(rgb_std)[:, None, None]
    state_dict[name + '.weight'] = w


def main(pytorch_model, out_model, adjust_input=False):
    layers = [3, 4, 23, 3]
    block_depth = 3
    # prefix = 'backbone.'
    prefix = ''
    # weights = torch.load(pytorch_model)['state_dict']
    weights = model_zoo.load_url(model_urls['resnet101'])
    from IPython import embed
    embed()
    fout = h5py.File(out_model, 'w')
    # if adjust_input:
    #     adjust_input_layer(state_dict, 'conv1')
    convert_conv(weights, prefix+'conv1', fout, 'conv1')
    convert_bn(weights, prefix+'bn1', fout, 'bn1')
    for i in range(len(layers)):
        for k in range(1, block_depth + 1):
            convert_conv(weights, prefix+'layer{}.0.conv{}'.format(i + 1, k), fout, 'res{}a.conv{}'.format(i + 2, k))
            convert_bn(weights, prefix+'layer{}.0.bn{}'.format(i + 1, k), fout, 'res{}a.bn{}'.format(i + 2, k))
        if prefix+'layer{}.0.downsample.0.weight'.format(i+1) in weights.keys():
            convert_conv(weights, prefix+'layer{}.0.downsample.0'.format(i + 1), fout, 'res{}a.shortcut'.format(i + 2))
            convert_bn(weights, prefix+'layer{}.0.downsample.1'.format(i + 1), fout, 'res{}a.shortcut_bn'.format(i + 2))
        for j in range(1, layers[i]):
            for k in range(1, block_depth + 1):
                convert_conv(weights, prefix+'layer{}.{}.conv{}'.format(
                                i + 1, j, k), fout, 'res{}b{}.conv{}'.format(i + 2, j, k))
                convert_bn(weights, prefix+'layer{}.{}.bn{}'.format(i + 1, j, k), fout, 'res{}b{}.bn{}'.format(i + 2, j, k))
    # convert_conv(weights, 'rpn_top.rpn_conv', fout, 'rpn_output')
    # convert_conv(weights, 'rpn_top.rpn_cls_score', fout,'rpn_cls_score')
    # convert_conv(weights, 'rpn_top.rpn_bbox_pred', fout,'rpn_bbox_pred')
    # convert_conv(weights, 'conv_new', fout, 'conv_new')
    # convert_conv(weights, 'conv_rfcn_cls', fout, 'conv_rfcn_cls')
    # convert_conv(weights, 'conv_rfcn_reg', fout, 'conv_rfcn_reg')
    fout.close()

if __name__ == '__main__':
    main("/home/wjq/rfcn_latest.pth", '/mnt/gv7/wjq/pytorch2parrtos/pytorch2parrots_resnet101_v1.parrots')
