from __future__ import absolute_import
import os.path as osp

from PIL import Image
from lz import *


class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.cache = dict()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath2 = '/home/xinglu/prj/openpose/cu.out/' + fname.split('.')[0] + '_rendered.png'
        fpath3 = '/home/xinglu/.torch/data/cuhk03/label/npy/' + fname.split('.')[0] + '.npy'
        fpath = fname
        if not osp.exists(fpath) and self.root is not None:
            fpath = osp.join(self.root, fname)
        if fpath in self.cache:
            (img, npy, fname, pid, camid) = self.cache[fpath]
            img = self.transform(img)
            return img, npy, fname, pid, camid
        img = Image.open(fpath).convert('RGB')
        npy = np.load(fpath3)
        npy = to_torch(npy)
        self.cache[fpath] = (img, npy, fname, pid, camid)
        if self.transform is not None:
            img = self.transform(img)
        # self.transform.transforms[0].use_last = True
        # self.transform.transforms[1].use_last=True
        # pose = Image.open(fpath2).convert('RGB')
        # pose = self.transform(pose)
        # img = torch.cat([img,pose])

        return img, npy, fname, pid, camid


class KeyValuePreprocessor(object):
    def __init__(self, dataset):
        super(KeyValuePreprocessor, self).__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, keys):
        if isinstance(keys, (tuple, list)):
            return [self.dataset[key] for key in keys]
        return self.dataset[keys]


class IndValuePreprocessor(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset.size(0)

    def __getitem__(self, keys):
        if isinstance(keys, (tuple, list)):
            return [self.dataset[key, :] for key in keys]
        return self.dataset[keys]
