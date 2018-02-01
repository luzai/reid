from lz import *

# python examples/triplet_loss.py -d cuhk03 -a resnet50 --combine-trainval --logs-dir examples/logs/triplet-loss/cuhk03-resnet50

# python examples/softmax_loss.py -d cuhk03 -a resnet50 --combine-trainval --logs-dir examples/logs/softmax-loss/cuhk03-resnet50


conf = edict(
    arc='resnet50',
    logs_dir='work/triplet',
    ndevs=2,
)


conf = edict(
    arc='resnet50',
    logs_dir='work/softmax',
    ndevs=2,
)