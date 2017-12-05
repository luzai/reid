from unittest import TestCase
from lz import *

class TestPreprocessor(TestCase):
    def test_getitem(self):
        import torchvision.transforms as t
        from reid.datasets.viper import VIPeR
        from reid.utils.data.preprocessor import Preprocessor

        root, split_id, num_val = '/home/xinglu/.torch/data/viper', 0, 100
        dataset = VIPeR(root, split_id=split_id, num_val=num_val, download=True)

        preproc = Preprocessor(dataset.train, root=dataset.images_dir,
                               transform=t.Compose([
                                   t.Scale(256),
                                   t.CenterCrop(224),
                                   t.ToTensor(),
                                   t.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                               ]))
        self.assertEquals(len(preproc), len(dataset.train))
        img, fname, pid, camid = preproc[0]
        self.assertEquals(img.size(), (3, 224, 224))


class TestRandomIdentitySampler(TestCase):
    def test_iter(self):
        from reid.utils.data.sampler import RandomIdentitySampler
        from reid.datasets.cuhk03 import CUHK03
        from collections import defaultdict

        root, split_id, num_val = '/home/xinglu/.torch/data/cuhk03', 0, 100
        dataset = CUHK03(root, split_id=split_id, num_val=num_val, download=True)
        tran_set = dataset.trainval
        sampler = RandomIdentitySampler(tran_set, 4, 12, shuffle=True)

        index_dic = defaultdict(list)
        ind2pid = dict()
        for index, (_, pid, _) in enumerate(tran_set):
            index_dic[pid].append(index)
            ind2pid[index] = pid
        res = list(sampler.__iter__())[:12]
        res = [ind2pid[res_] for res_ in res]
        print(res)
        self.assertEquals(np.unique(res).shape[0], 3)

