from __future__ import absolute_import
import os.path as osp

from PIL import Image
from lz import *


class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None, test_aug=False, has_npy=False, has_pose=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.cache = dict()
        self.test_aug = test_aug  # todo
        self.has_npy = has_npy
        self.has_pose = has_pose

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        res = {}
        fname, pid, cid = self.dataset[index]
        res['fname'] = fname
        res['pid'] = pid
        res['cid'] = cid

        fpath2 = '/home/xinglu/prj/openpose/cu.out/' + fname.split('.')[0] + '_rendered.png'
        fpath3 = '/home/xinglu/.torch/data/cuhk03/label/npy/' + fname.split('.')[0] + '.npy'
        fpath = fname
        if not osp.exists(fpath) and self.root is not None:
            fpath = osp.join(self.root, fpath)
        if fpath in self.cache:
            res = self.cache[fpath]
            img = self.transform(res['img'])
            res_return = copy.deepcopy(res)
            res_return.update({'img': img})
            return res_return

        res['img'] = Image.open(fpath).convert('RGB')
        if self.has_npy:
            res['npy'] = to_torch(np.load(fpath3))
        self.cache[fpath] = res
        img = self.transform(res['img'])
        res_return = copy.deepcopy(res)
        res_return.update({'img': img})
        return res_return


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
