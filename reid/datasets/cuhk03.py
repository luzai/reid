from __future__ import print_function, absolute_import
import os.path as osp

import numpy as np

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json,read_json


def _pluck(identities, indices, relabel=False):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                name = osp.splitext(fname)[0]
                x, y, _ = map(int, name.split('_'))
                assert pid == x and camid == y
                if relabel:
                    ret.append((fname, index, camid))
                else:
                    ret.append((fname, pid, camid))
    return ret

class CUHK03(Dataset):
    url = 'https://docs.google.com/spreadsheet/viewform?usp=drive_web&formkey=dHRkMkFVSUFvbTJIRkRDLWRwZWpONnc6MA#gid=0'
    md5 = '728939e58ad9f0ff53e521857dd8fb43'

    def _check_integrity(self):
        if self.mode !='':
            return osp.isdir(osp.join(self.root, 'detect')) and \
                    osp.isdir(osp.join(self.root, 'label')) and \
                    osp.isdir(osp.join(self.root, 'combine'))

        else:
            return osp.isdir(osp.join(self.root, 'images')) and \
                   osp.isfile(osp.join(self.root, 'meta.json')) and \
                   osp.isfile(osp.join(self.root, 'splits.json'))

    def load(self, num_val=0.3, verbose=True):
        splits = read_json(osp.join(self.root, self.mode,  'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]

        # Randomly split train / val
        trainval_pids = np.asarray(self.split['trainval'])
        np.random.shuffle(trainval_pids)
        num = len(trainval_pids)
        if isinstance(num_val, float):
            num_val = int(round(num * num_val))
        if num_val >= num or num_val < 0:
            raise ValueError("num_val exceeds total identities {}"
                             .format(num))
        train_pids = sorted(trainval_pids[:-num_val])
        val_pids = sorted(trainval_pids[-num_val:])

        self.meta = read_json(osp.join(self.root,self.mode,  'meta.json'))
        identities = self.meta['identities']
        self.train = _pluck(identities, train_pids, relabel=True)
        self.val = _pluck(identities, val_pids, relabel=True)
        self.trainval = _pluck(identities, trainval_pids, relabel=True)
        self.query = _pluck(identities, self.split['query'])
        self.gallery = _pluck(identities, self.split['gallery'])
        self.num_train_ids = len(train_pids)
        self.num_val_ids = len(val_pids)
        self.num_trainval_ids = len(trainval_pids)

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
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

    def __init__(self, root, split_id=0, num_val=100, download=True, mode=''):
        super(CUHK03, self).__init__(root, split_id=split_id)
        self.mode = mode
        if download:
            self.download()

        if not self._check_integrity():

            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)

    def download(self):
        if self._check_integrity():
        # if False:

            print("Files already downloaded and verified")
            return

        import h5py
        import hashlib
        from scipy.misc import imsave
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'cuhk03_release.zip')

        if osp.isfile(fpath) and \
                        hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.md5:
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually from {} "
                               "to {}".format(self.url, fpath))

        # # Extract the file
        exdir = osp.join(raw_dir, 'cuhk03_release')
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=raw_dir)

        # Format
        images_dir = osp.join(self.root, self.mode, 'images')

        mkdir_if_missing(images_dir)
        matdata = h5py.File(osp.join(exdir, 'cuhk-03.mat'), 'r')

        def deref(ref):
            return matdata[ref][:].T

        def dump_(refs, pid, cam, fnames):
            for ref in refs:
                img = deref(ref)
                if img.size == 0 or img.ndim < 2: break
                fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, cam, len(fnames))
                imsave(osp.join(images_dir, fname), img)
                fnames.append(fname)

        identities = []
        for labeled, detected in zip(
                matdata['labeled'][0], matdata['detected'][0]):
            labeled, detected = deref(labeled), deref(detected)
            assert labeled.shape == detected.shape
            for i in range(labeled.shape[0]):
                pid = len(identities)
                images = [[], []]
                if self.mode == 'combine':
                    dump_(labeled[i, :5], pid, 0, images[0])
                    dump_(detected[i, :5], pid, 0, images[0])
                    dump_(labeled[i, 5:], pid, 1, images[1])
                    dump_(detected[i, 5:], pid, 1, images[1])
                elif self.mode == 'label' or self.mode == '':
                    dump_(labeled[i, :5], pid, 0, images[0])
                    dump_(labeled[i, 5:], pid, 1, images[1])
                elif self.mode == 'detect':
                    dump_(detected[i, :5], pid, 0, images[0])
                    dump_(detected[i, 5:], pid, 1, images[1])
                identities.append(images)

        # Save meta information into a json file
        meta = {'name': 'cuhk03', 'shot': 'multiple', 'num_cameras': 2,
                'identities': identities}
        write_json(meta, osp.join(self.root, self.mode, 'meta.json'))

        # Save training and test splits
        splits = []
        view_counts = [deref(ref).shape[0] for ref in matdata['labeled'][0]]
        vid_offsets = np.r_[0, np.cumsum(view_counts)]
        for ref in matdata['testsets'][0]:
            test_info = deref(ref).astype(np.int32)
            test_pids = sorted(
                [int(vid_offsets[i - 1] + j - 1) for i, j in test_info])
            trainval_pids = list(set(range(vid_offsets[-1])) - set(test_pids))
            split = {'trainval': trainval_pids,
                     'query': test_pids,
                     'gallery': test_pids}
            splits.append(split)
        write_json(splits, osp.join(self.root, self.mode, 'splits.json'))
