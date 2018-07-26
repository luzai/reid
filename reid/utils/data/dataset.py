from __future__ import print_function
import os
import os.path as osp
import numpy as np

from reid.utils.serialization import read_json
import lz


def _pluck(identities, indices, relabel=False,root=''):
    pid2lbl = {}
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                # name = osp.splitext(fname)[0]
                # x, y, _ = map(int, name.split('_'))
                # assert pid == x and camid == y
                if not osp.exists(fname):
                    fname = f'{root}/images/{fname}'
                assert osp.exists(fname)
                if relabel:
                    ret.append((fname, index, camid))
                    pid2lbl[pid] = index
                else:
                    ret.append((fname, pid, camid))
    if relabel:
        return ret, pid2lbl
    else:
        return ret


def _stats(tensor):
    tensor = np.asarray(tensor)
    return tensor.shape, np.unique(tensor).shape, tensor.max()


class Dataset(object):
    def __init__(self, root, split_id=0):
        self.root = root
        self.split_id = split_id
        self.meta = None
        self.split = None
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0
        self.images_dir = osp.join(self.root, 'images')

    def load(self, num_val=0.3, verbose=True):
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]

        # Randomly split train / val
        trainval_pids = np.asarray(self.split['trainval'])
        # np.random.shuffle(trainval_pids) # because we use trainval as train, val to visualization
        num = len(trainval_pids)
        if isinstance(num_val, float):
            num_val = int(round(num * num_val))
        if num_val >= num or num_val < 0:
            raise ValueError("num_val exceeds total identities {}"
                             .format(num))
        train_pids = sorted(trainval_pids[:-num_val])
        val_pids = sorted(trainval_pids[-num_val:])

        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']
        self.train, _ = _pluck(identities, train_pids, relabel=True, root = self.root)
        self.val, _ = _pluck(identities, val_pids, relabel=True, root = self.root)
        self.trainval, pid2lbl = _pluck(identities, trainval_pids, relabel=True, root = self.root)
        self.pid2lbl = pid2lbl
        self.query = _pluck(identities, self.split['query'], root = self.root)
        self.gallery = _pluck(identities, self.split['gallery'], root = self.root)
        self.num_train_ids = len(train_pids)
        self.num_val_ids = len(val_pids)
        self.num_trainval_ids = len(trainval_pids)

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  --------|--------|-----------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_ids, len(self.train)))
            print("  val      | {:5d} | {:8d}"
                  .format(self.num_val_ids, len(self.val)))
            print("  trainval | {:5d} | {:8d}"
                  .format(self.num_trainval_ids, len(self.trainval)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(self.split['query']), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(self.split['gallery']), len(self.gallery)))

    def _check_integrity(self):
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json'))
