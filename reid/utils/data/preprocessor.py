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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath2 = '/data1/xinglu/prj/openpose/out/' + fname.split('.')[0]+'_rendered.png'
        fpath = fname
        if not osp.exists(fpath) and self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        self.transform.transforms[0].use_last=True
        # self.transform.transforms[1].use_last=True
        # pose = Image.open(fpath2).convert('RGB')
        # pose = self.transform(pose)
        # img = torch.cat([img,pose])
        return img, fname, pid, camid


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
