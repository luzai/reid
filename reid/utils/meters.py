from __future__ import absolute_import


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # if type(val).__module__ == 'numpy':
        #     if val.shape == ():
        #         val = float(val)
        #     else:
        #         val = float(val[0])
        try:
            val = float(val)
        except Exception as inst:
            print(inst)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
