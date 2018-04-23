import numpy as np


class DopInfo(object):
    def __init__(self, num_classes):
        self.dop = np.ones(num_classes, dtype=np.int64) * -1
