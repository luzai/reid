import torch
from lz import *


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):
        if len(dist.size()) == 2:
            dist = dist[:, 0]
        loss_contrastive = 1 / 2. * torch.mean(
            (label.float()) * torch.pow(dist, 2) +
            (1. - label.float()) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        )

        return loss_contrastive
