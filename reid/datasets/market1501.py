from __future__ import print_function, absolute_import
import os.path as osp

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json
from lz import *
from scipy.io import loadmat


class Mars(object):
    """
    MARS

    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.

    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
    root = '/data1/xinglu/work/data/mars'
    train_name_path = osp.join(root, 'info/train_name.txt')
    test_name_path = osp.join(root, 'info/test_name.txt')
    track_train_info_path = osp.join(root, 'info/tracks_train_info.mat')
    track_test_info_path = osp.join(root, 'info/tracks_test_info.mat')
    query_IDX_path = osp.join(root, 'info/query_IDX.mat')

    def __init__(self, min_seq_len=0, **kwargs):
        self._check_before_run()

        if osp.exist(self.root+'/cache.h5'):
            with Database(self.root + '/cache.h5') as db:
                # print(list((db.keys())))
                num_train_pids, num_query_pids, num_gallery_pids = \
                    db['num_train_pids'], db['num_query_pids'], db[
                        'num_gallery_pids']
                num_train_pids, num_query_pids, num_gallery_pids = map(int, (num_train_pids,
                                                                             num_query_pids,
                                                                             num_gallery_pids))
            with pd.HDFStore(self.root + '/cache.h5') as db:
                train, query, gallery = db['train'], db['query'], db['gallery']
                train, query, gallery = map(
                    lambda train: pd.DataFrame(train).to_records(index=False).tolist(),
                    (train, query, gallery)
                )

        else:
            # prepare meta data
            train_names = self._get_names(self.train_name_path)
            test_names = self._get_names(self.test_name_path)
            track_train = loadmat(self.track_train_info_path)['track_train_info']  # numpy.ndarray (8298, 4)
            track_test = loadmat(self.track_test_info_path)['track_test_info']  # numpy.ndarray (12180, 4)
            query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze()  # numpy.ndarray (1980,)
            query_IDX -= 1  # index from 0
            track_query = track_test[query_IDX, :]
            gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
            track_gallery = track_test[gallery_IDX, :]

            train, num_train_tracklets, num_train_pids, num_train_imgs = \
                self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

            query, num_query_tracklets, num_query_pids, num_query_imgs = \
                self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

            gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
                self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

            num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
            min_num = np.min(num_imgs_per_tracklet)
            max_num = np.max(num_imgs_per_tracklet)
            avg_num = np.mean(num_imgs_per_tracklet)

            num_total_pids = num_train_pids + num_query_pids
            num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

            print("=> MARS loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # tracklets")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
            print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
            print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
            print("  ------------------------------")

            with Database(self.root + '/cache.h5') as db:
                db['num_train_pids'], db['num_query_pids'], db[
                    'num_gallery_pids'], = num_train_pids, num_query_pids, num_gallery_pids
                db.flush()

            with pd.HDFStore(self.root + '/cache.h5') as db:
                train = pd.DataFrame(train)
                query = pd.DataFrame(query)
                gallery = pd.DataFrame(gallery)
                db['train'], db['query'], db['gallery'] = train, query, gallery
                db.flush()

        self.train = train
        self.val = None
        self.trainval = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_val_pids = 0
        self.num_trainval_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1  # index starts from 0
            img_names = names[start_index - 1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class Market1501(Dataset):
    url = 'https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view'
    md5 = '65005ab7d12ec1c44de4eeafe813e68a'

    root = '/data1/xinglu/work/data/market1501/'
    train_dir = osp.join(root, 'bounding_box_train')
    query_dir = osp.join(root, 'query')
    gallery_dir = osp.join(root, 'bounding_box_test')

    def __init__(self, root=None, split_id=0, num_val=100, download=True, check_intergrity=True, **kwargs):
        super(Market1501, self).__init__(root, split_id=split_id)
        self.root = '/data1/xinglu/work/data/market1501/'

        # if download:
        #     self.download(check_intergrity)
        #
        # if not self._check_integrity():
        #     raise RuntimeError("Dataset not found or corrupted. " +
        #                        "You can use download=True to download it.")
        #
        # self.load(num_val)
        #
        # return

        self._check_before_run()
        if osp.exists(self.root + '/cache.h5'):
            with Database(self.root + '/cache.h5') as db:
                # print(list((db.keys())))
                num_train_pids, num_query_pids, num_gallery_pids = \
                    db['num_train_pids'], db['num_query_pids'], db[
                        'num_gallery_pids']
                num_train_pids, num_query_pids, num_gallery_pids = map(int, (num_train_pids,
                                                                             num_query_pids,
                                                                             num_gallery_pids))
            with pd.HDFStore(self.root + '/cache.h5') as db:
                train, query, gallery = db['train'], db['query'], db['gallery']
                train, query, gallery = map(
                    lambda train: pd.DataFrame(train).to_records(index=False).tolist(),
                    (train, query, gallery)
                )

        else:
            train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
            query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
            gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
            num_total_pids = num_train_pids + num_query_pids
            num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

            print("=> Market1501 loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # images")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
            print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
            print("  ------------------------------")
            with Database(self.root + '/cache.h5') as db:
                db['num_train_pids'], db['num_query_pids'], db[
                    'num_gallery_pids'], = num_train_pids, num_query_pids, num_gallery_pids
                db.flush()

            with pd.HDFStore(self.root + '/cache.h5') as db:
                train = pd.DataFrame(train)
                query = pd.DataFrame(query)
                gallery = pd.DataFrame(gallery)
                db['train'], db['query'], db['gallery'] = train, query, gallery
                db.flush()

        self.train = train
        self.val = None
        self.trainval = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_trainval_ids = num_train_pids
        self.num_val_ids = 0
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs

    def download(self, check_integrity=True):
        if check_integrity and self._check_integrity():
            print("Files already downloaded and verified")
            return

        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'Market-1501-v15.09.15.zip')
        if osp.isfile(fpath) and \
                hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.md5:
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually from {} "
                               "to {}".format(self.url, fpath))

        # Extract the file
        exdir = osp.join(raw_dir, 'Market-1501-v15.09.15')
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=raw_dir)

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        # 1501 identities (+1 for background) with 6 camera views each
        identities = [[[] for _ in range(6)] for _ in range(1502)]

        def register(subdir, pattern=re.compile(r'([-\d]+)_c(\d)')):
            fnames = []
            fpaths = sorted(glob(osp.join(exdir, subdir, '*.jpg')))
            pids = set()
            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid, cam = map(int, pattern.search(fname).groups())
                if pid == -1: continue  # junk images are just ignored
                assert 0 <= pid <= 1501  # pid == 0 means background
                assert 1 <= cam <= 6
                cam -= 1
                pids.add(pid)
                fname = ('{:08d}_{:02d}_{:04d}.jpg'
                         .format(pid, cam, len(identities[pid][cam])))
                fnames.append(fname)
                identities[pid][cam].append(fname)
                shutil.copy(fpath, osp.join(images_dir, fname))
            return pids, fnames

        trainval_pids, _ = register('bounding_box_train')
        gallery_pids, fnames = register('bounding_box_test')
        # cvb.dump(fnames, work_path+'/mk.gallery.pkl')
        query_pids, fnames = register('query')
        # cvb.dump(fnames, work_path + '/mk.query.pkl')
        assert query_pids <= gallery_pids
        assert trainval_pids.isdisjoint(gallery_pids)

        # Save meta information into a json file
        meta = {'name': 'Market1501', 'shot': 'multiple', 'num_cameras': 6,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))

# if __name__ == '__main__':
#     Market1501()
#     Mars()
