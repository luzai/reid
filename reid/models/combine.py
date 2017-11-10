from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
import subprocess


class Combine(nn.Module):
    def __init__(self, model, transform, match):
        super(Combine, self).__init__()
        self.model = model
        self.tranform = transform
        self.match = match

    def forward(self, inputs, targets):
        outputs = self.model(inputs)
        inputs[0].size(), outputs.size()
        pair1, pair2, y = self.tranform(outputs, targets)
        pred = self.match(pair1, pair2)
        return pred, y