from __future__ import absolute_import
from collections import OrderedDict

from torch.autograd import Variable
import torch
from ..utils.meters import *
from ..utils import to_numpy, to_torch
import time


def extract_cnn_feature(model, inputs, modules=None, gpu=(0,)):
    model.eval()
    inputs = to_torch(inputs)
    if not torch.cuda.is_available():
        inputs = Variable(inputs, volatile=True)
    else:
        inputs = Variable(inputs, volatile=True).cuda()

    if modules is None:
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        outputs = outputs.data.cpu()
        return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None

        def func(m, i, o): outputs[id(m)] = o.data.cpu()

        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())


def extract_cnn_embeddings(model, inputs, modules=None):
    model.eval()

    for ind, inp in enumerate(inputs):
        inputs[ind] = to_torch(inp)
    inputs = [Variable(x, volatile=True).cuda() for x in inputs]

    assert modules is None

    outputs = model(*inputs)
    outputs = outputs.data.cpu()
    return outputs
