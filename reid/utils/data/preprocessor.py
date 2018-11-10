from __future__ import absolute_import
from PIL import Image
from lz import *
# from lomo.lomo_map import extract_feature
from torch.utils.data import Dataset
import copy
from itertools import chain


class DirPreprocessor(object):
    def __init__(self, dir, transorm):
        self.dir = dir
        self.imgs = glob.glob(dir + '/*.jpg') + \
                    glob.glob(dir + '/*.png') + \
                    glob.glob(dir + '/*.JPG')

        self.transform = transorm

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        imgp = self.imgs[index]
        return self.transform(read_image(imgp)), imgp


class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None, test_aug=False, has_npy=False, has_pose=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.cache = dict()
        self.test_aug = test_aug  # todo test time multi crop
        self.has_npy = has_npy
        self.has_pose = has_pose

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        res = self._get_single_item(indices)
        for k, v in res.items():
            assert (
                    isinstance(v, np.ndarray) or
                    isinstance(v, str) or
                    isinstance(v, int) or
                    isinstance(v, np.int64) or
                    torch.is_tensor(v)
            ), type(v)
        return res

    def _get_single_item(self, index):
        res = {}
        fname, pid, cid = self.dataset[index]
        res['fname'] = fname
        res['pid'] = pid
        res['cid'] = cid

        fpath = fname
        # if np.random.rand()>.5:
        #     fpath = '/home/xinglu/Nextcloud/1.jpg'
        # else:
        #     fpath = '/home/xinglu/Nextcloud/2.jpg'

        # if not osp.exists(fpath) and self.root is not None:
        #     fpath = osp.join(self.root, fpath)
        #
        # if fpath in self.cache:
        #     res = self.cache[fpath]
        #     img = self.transform(res['img'])
        #     res_return = copy.deepcopy(res)
        #     res_return.update({'img': img})
        #     return res_return
        if osp.exists(fpath) and os.stat(fpath).st_size == 0:
            rm(fpath)
            return self._get_single_item(0)
        res['img'] = read_image(fpath)
        # self.cache[fpath] = res
        img = self.transform(res['img'])
        # img = img.type(torch.double)
        if self.has_npy:
            # res['npy'] = to_torch(np.load(fpath3))
            raise NotImplementedError('do not use lomo cv2')
            # res['npy'] = extract_feature(img, fpath)
        res_return = copy.deepcopy(res)
        res_return.update({'img': img})
        return res_return


import lmdb, six

serielized = {}


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    if 'mars' in img_path:
        if 'mars' not in serielized:
            root = '/home/xinglu/.torch/data/mars/img_lmdb'
            osp.exists(root)
            env = lmdb.open(root, max_readers=1, readonly=True, lock=False,
                            readahead=False, meminit=False)
            serielized['mars'] = env
        else:
            env = serielized['mars']
        with env.begin(write=False) as txn:
            imgbuf = txn.get(osp.basename(img_path).encode())
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        # with Image.open(buf) as f:
        f = Image.open(buf)
        img = f.convert('RGB')
        assert img is not None
        return img

    got_img = False
    assert osp.exists(img_path), ("{} does not exist".format(img_path))
    while not got_img:
        try:
            # with Image.open(img_path) as f:
            f = Image.open(img_path)
            img = f.convert('RGB')
            assert img is not None
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass

    return img


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample == 'random':
            """
            Randomly sample seq_len items from num items,
            if num is smaller than seq_len, then replicate items
            """
            indices = np.arange(num)
            replace = False if num >= self.seq_len else True
            indices = np.random.choice(indices, size=self.seq_len, replace=replace)
            # sort indices to keep temporal order
            # comment it to be order-agnostic
            indices = np.sort(indices)
        elif self.sample == 'evenly':
            """Evenly sample seq_len items from num items."""
            if num >= self.seq_len:
                num -= num % self.seq_len
                indices = np.arange(0, num, num // self.seq_len)
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32) * (num - 1)])
            assert len(indices) == self.seq_len
        elif self.sample == 'all':
            """
            Sample all items, seq_len is useless now and batch_size needs
            to be set to 1.
            """
            indices = np.arange(num)
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

        imgs = []
        for index in indices:
            img_path = img_paths[index]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
        res = {'img': imgs, 'pid': pid, 'cid': camid, 'fname': img_paths}
        return res


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
