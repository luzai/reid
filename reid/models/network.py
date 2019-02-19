from .resnetori import *
import torch.nn as nn
import torch
from .globalNet import globalNet
from .refineNet import refineNet

__all__ = ['CPN50', 'CPN101']


class CPN(nn.Module):
    def __init__(self, resnet, output_shape, num_class, pretrained=True, args=None):
        super(CPN, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        self.resnet = resnet
        self.global_net = globalNet(channel_settings, output_shape, num_class=num_class, )
        self.refine_net = refineNet(channel_settings[-1], output_shape, num_class=num_class, args=args)
    
    def forward(self, x, ft=False, *args, **kwargs):
        # if len(x.shape) == 3:
        #     x = x.view(1, 3, 256, 128)
        if ft:
            torch.set_grad_enabled(False)
        res_out = self.resnet(x)
        if ft:
            torch.set_grad_enabled(True)
        global_fms, global_outs = self.global_net(res_out, ft)
        refine_out = self.refine_net(global_fms, ft)
        return global_outs, refine_out # todo
        # return refine_out


from lz import load_state_dict

def CPN50(out_size=(96, 72), num_class=128, pretrained=True, args=None, **kwargs):
    res50 = resnet50(pretrained=pretrained)
    model = CPN(res50, output_shape=out_size, num_class=num_class, pretrained=pretrained, args=args)
    if pretrained:
        state_dict = torch.load('/home/xinglu/work/reid/work.use/checkpoint/CPN50_384x288.pth.tar')[
            'state_dict']
        load_state_dict(model, state_dict, de_prefix='module.') # todo
    return model


def CPN101(out_size=(96, 72), num_class=128, pretrained=True, args=None, **kwargs):
    res101 = resnet101(pretrained=pretrained)
    model = CPN(res101, output_shape=out_size, num_class=num_class, pretrained=pretrained, args=args)
    return model
