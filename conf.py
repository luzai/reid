from lz import *

conf = edict(
    arc='resnet50',
    logs_dir='work/triplet',
    ndevs=2,
)

conf = edict(
    arc='resnet50',
    logs_dir='work/bak',
    ndevs=1,
    batch_size=128
)
