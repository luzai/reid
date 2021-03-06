from __future__ import absolute_import
from collections import OrderedDict
from lz import *
from ..utils.meters import *


def extract_cnn_feature(model, inputs, modules=None):
    model.eval()
    with torch.no_grad():
        if modules is None:
            outputs = model(*inputs)
            if isinstance(outputs, collections.Sequence):
                outputs = outputs[-1]
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
    with torch.no_grad():
        inputs = [x.cuda() for x in inputs]

        assert modules is None

        outputs = model(*inputs)
        outputs = outputs.data.cpu()
        return outputs
