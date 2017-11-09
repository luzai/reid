from __future__ import absolute_import

from .oim import oim, OIM, OIMLoss
from .triplet import TripletLoss
from .tuplet import TupletLoss
from .learnable import LearnableLoss

__all__ = [
    'oim',
    'OIM',
    'OIMLoss',
    'TripletLoss',
    'TupletLoss',
    'LearnableLoss'
]
