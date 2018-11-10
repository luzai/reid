from __future__ import print_function, absolute_import
import os.path as osp

from reid.utils.data import Dataset
from reid.utils.serialization import write_json, read_json
import tarfile
from lz import *
import lz
import urllib
from scipy.io import loadmat


class Mars(object):
    """
    MARS

    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.

    URL: http://www.liangzheng.com.cn/Project/project_mars.html

    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)
    # cameras: 6
    """
    dataset_dir = 'mars'

    def __init__(self, root='/home/xinglu/.torch/data', min_seq_len=0, **kwargs):
        root = '/home/xinglu/.torch/data'
        self.images_dir = self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_name_path = osp.join(self.dataset_dir, 'info/train_name.txt')
        self.test_name_path = osp.join(self.dataset_dir, 'info/test_name.txt')
        self.track_train_info_path = osp.join(self.dataset_dir, 'info/tracks_train_info.mat')
        self.track_test_info_path = osp.join(self.dataset_dir, 'info/tracks_test_info.mat')
        self.query_IDX_path = osp.join(self.dataset_dir, 'info/query_IDX.mat')

        self._check_before_run()

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

        self.train = train
        self.trainval = train
        self.val = None
        self.query = query
        self.gallery = gallery

        self.num_train_pids = self.num_trainval_pids = self.num_train_ids = self.num_trainval_ids = num_train_pids
        self.num_val_ids = self.num_val_pids = 0
        self.num_query_ids = self.num_query_pids = num_query_pids
        self.num_gallery_ids = self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
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
            img_paths = [osp.join(self.dataset_dir, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class iLIDSVID(object):
    """
    iLIDS-VID

    Reference:
    Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.

    URL: http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html

    Dataset statistics:
    # identities: 300
    # tracklets: 600
    # cameras: 2
    """
    dataset_dir = 'ilids-vid'

    def __init__(self, root='/home/xinglu/.torch/data/', split_id=0, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_url = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'
        self.data_dir = osp.join(self.dataset_dir, 'i-LIDS-VID')
        self.split_dir = osp.join(self.dataset_dir, 'train-test people splits')
        self.split_mat_path = osp.join(self.split_dir, 'train_test_splits_ilidsvid.mat')
        self.split_path = osp.join(self.dataset_dir, 'splits.json')
        self.cam_1_path = osp.join(self.dataset_dir, 'i-LIDS-VID/sequences/cam1')
        self.cam_2_path = osp.join(self.dataset_dir, 'i-LIDS-VID/sequences/cam2')

        self._download_data()
        self._check_before_run()

        self._prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                "split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits) - 1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
            self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
            self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
            self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> iLIDS-VID loaded")
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

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _download_data(self):
        if osp.exists(self.dataset_dir):
            print("This dataset has been downloaded.")
            return

        mkdir_p(self.dataset_dir, delete=False)
        fpath = osp.join(self.dataset_dir, osp.basename(self.dataset_url))

        print("Downloading iLIDS-VID dataset")
        url_opener = urllib.URLopener()
        url_opener.retrieve(self.dataset_url, fpath)

        print("Extracting files")
        tar = tarfile.open(fpath)
        tar.extractall(path=self.dataset_dir)
        tar.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError("'{}' is not available".format(self.split_dir))

    def _prepare_split(self):
        if not osp.exists(self.split_path):
            print("Creating splits")
            mat_split_data = loadmat(self.split_mat_path)['ls_set']

            num_splits = mat_split_data.shape[0]
            num_total_ids = mat_split_data.shape[1]
            assert num_splits == 10
            assert num_total_ids == 300
            num_ids_each = num_total_ids // 2

            # pids in mat_split_data are indices, so we need to transform them
            # to real pids
            person_cam1_dirs = os.listdir(self.cam_1_path)
            person_cam2_dirs = os.listdir(self.cam_2_path)

            # make sure persons in one camera view can be found in the other camera view
            assert set(person_cam1_dirs) == set(person_cam2_dirs)

            splits = []
            for i_split in range(num_splits):
                # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
                train_idxs = sorted(list(mat_split_data[i_split, num_ids_each:]))
                test_idxs = sorted(list(mat_split_data[i_split, :num_ids_each]))

                train_idxs = [int(i) - 1 for i in train_idxs]
                test_idxs = [int(i) - 1 for i in test_idxs]

                # transform pids to person dir names
                train_dirs = [person_cam1_dirs[i] for i in train_idxs]
                test_dirs = [person_cam1_dirs[i] for i in test_idxs]

                split = {'train': train_dirs, 'test': test_dirs}
                splits.append(split)

            print("Totally {} splits are created, following Wang et al. ECCV'14".format(len(splits)))
            print("Split file is saved to {}".format(self.split_path))
            write_json(splits, self.split_path)

        print("Splits created")

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname: i for i, dirname in enumerate(dirnames)}

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class PRID(object):
    """
    PRID

    Reference:
    Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.

    URL: https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/

    Dataset statistics:
    # identities: 200
    # tracklets: 400
    # cameras: 2
    """
    dataset_dir = 'prid2011'

    def __init__(self, root='/home/xinglu/.torch/data/', split_id=0, min_seq_len=0, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_url = 'https://files.icg.tugraz.at/f/6ab7e8ce8f/?raw=1'
        self.split_path = osp.join(self.dataset_dir, 'splits_prid2011.json')
        self.cam_a_path = osp.join(self.dataset_dir, 'prid_2011', 'multi_shot', 'cam_a')
        self.cam_b_path = osp.join(self.dataset_dir, 'prid_2011', 'multi_shot', 'cam_b')

        self._check_before_run()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                "split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits) - 1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
            self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
            self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
            self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> PRID-2011 loaded")
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

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname: i for i, dirname in enumerate(dirnames)}

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_b_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class Market1501(Dataset):
    url = 'https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view'
    md5 = '65005ab7d12ec1c44de4eeafe813e68a'

    root = '/data1/xinglu/work/data/market1501/'
    train_dir = osp.join(root, 'bounding_box_train')
    query_dir = osp.join(root, 'query')
    gallery_dir = osp.join(root, 'bounding_box_test')

    def __init__(self, root='/data1/xinglu/work/data/market1501/', split_id=0, num_val=100, download=True,
                 check_intergrity=True, **kwargs):
        self.name = 'market1501'
        super(Market1501, self).__init__(root, split_id=split_id)
        self.root = root

        train, num_train_pids, num_train_imgs, pid2lbl = self._process_dir(self.train_dir, relabel=True)
        # name = getattr(self, 'name', 'fk')
        # lz.pickle_dump(pid2lbl, f'{name}.pid2lbl.pkl')
        logging.info(f'pid2lbl {pid2lbl}')
        self.pid2lbl = pid2lbl
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
        if relabel:
            return dataset, num_pids, num_imgs, pid2label
        else:
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
        mkdir_p(raw_dir, False)

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
        mkdir_p(images_dir, delete=False)

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


# need to provide: num_train_pids = num_trainval_pids, num_query_pids, num_gallary_pids
# trainval=train, val=None, query, gallery
# format: imgp, pid, cid

class Extract(Dataset):
    def __init__(self, **kwargs):
        self.name = 'extract'
        # self.root = work_path + 'reid.person'
        self.root = work_path + 'extra_train'
        assert osp.exists(self.root)
        dftl = []
        for pid in range(10):
            imps = glob.glob(self.root + f'/{pid}/*')  # png or jpg
            dft = pd.DataFrame({'imgs': imps, 'pids': pid, 'cids': np.arange(len(imps))})
            dftl.append(dft)
        df = pd.concat(dftl, axis=0)
        all_ind = np.random.permutation(df.shape[0])

        traintest_split = None
        if traintest_split is not None:
            train_ind = all_ind[:df.shape[0] * (traintest_split - 1) // traintest_split]
            test_ind = all_ind[df.shape[0] * (traintest_split - 1) // traintest_split:]
            df_train = df.iloc[train_ind]
            df_test = df.iloc[test_ind]
            self.gallery = self.query = df_test.to_records(index=False).tolist()
            self.train = self.trainval = df_train.to_records(index=False).tolist()
        else:
            self.gallery = self.query = self.train = self.trainval = df.to_records(index=False).tolist()
        self.val = None
        self.num_train_pids = 10
        self.num_trainval_ids = 10
        self.num_val_ids = 0
        self.num_query_pids = 10
        self.num_gallery_pids = 10
        self.images_dir = None
        # for pid, dft in df.groupby('pids'):
        #     print(pid, dft.shape)
        print(len(self.trainval), len(self.gallery))
        pass


class CUB(Dataset):
    def __init__(self, **kwargs):
        self.name = 'cub'
        self.root = self.images_dir = '/data2/share/cub200_2011/'

        images = np.loadtxt(self.root + '/images.txt', dtype=object)[:, 1]
        splits = np.loadtxt(self.root + '/train_test_split.txt', dtype=int)
        images_cls = np.array(pd.DataFrame(images).iloc[:, 0].str.split('.').str.get(0).tolist(), dtype=int)
        train_cls = splits[splits[:, 1] == 1, 0]
        test_cls = splits[splits[:, 1] == 0, 1]
        self.num_trainval_ids = train_cls.shape[0]
        self.num_query_ids = test_cls.shape[0]
        self.num_gallery_ids = test_cls.shape[0]

        df = pd.DataFrame({'path': images, 'cls': images_cls, 'is_train': splits[:, 1]})
        df['cids'] = np.arange(df.shape[0])
        df['path'] = self.root + 'images/' + df.path
        # df.head()
        df_train = df[df.is_train == 1]
        df_test = df[df.is_train == 0]
        self.trainval = df_train[['path', 'cls', 'cids']].to_records(index=False).tolist()
        self.query = self.gallery = df_test[['path', 'cls', 'cids']].to_records(index=False).tolist()
        self.val = None
        self.train = self.trainval
        self.num_val_ids = 0
        self.num_train_ids = self.num_trainval_ids


class Stanford_Prod(Dataset):
    def __init__(self, **kwargs):
        self.name = 'stanford_prod'
        self.root = self.images_dir = '/data2/share/online_products/Stanford_Online_Products/'

        train_df = pd.read_csv(self.root + 'Ebay_train.txt', sep=' ')
        test_df = pd.read_csv(self.root + 'Ebay_test.txt', sep=' ')
        train_df['cids'] = np.arange(train_df.shape[0])
        train_df['path'] = self.root + train_df.path
        test_df['cids'] = np.arange(test_df.shape[0])
        test_df['path'] = self.root + test_df.path
        # train_df.head()
        self.num_trainval_ids = np.unique(train_df.class_id).shape[0]
        self.num_query_ids = self.num_gallery_ids = np.unique(test_df.class_id).shape[0]
        # self.trainval = train_df[['path', 'class_id', 'cids']][:128].to_records(index=False).tolist()
        # self.query = self.gallery = test_df[['path', 'class_id', 'cids']][:128].to_records(index=False).tolist()
        self.trainval = train_df[['path', 'class_id', 'cids']].to_records(index=False).tolist()
        self.query = self.gallery = test_df[['path', 'class_id', 'cids']].to_records(index=False).tolist()

        self.val = None
        self.train = self.trainval
        self.num_val_ids = 0
        self.num_train_ids = self.num_trainval_ids


class Car196(Dataset):
    def __init__(self, **kwargs):
        self.root = self.images_dir = '/data2/share/cars196/CARS196/'
        self.name = 'car196'
        self.anno_path = self.root + 'cars_annos.mat'
        cars_annos = loadmat(self.anno_path)
        annotations = cars_annos["annotations"].ravel()
        self.img_paths = [str(anno[0][0]) for anno in annotations]
        self.img_paths = np.asarray(self.img_paths)
        self.labels = [int(anno[5]) for anno in annotations]
        df = pd.DataFrame({'path': self.img_paths, 'pids': self.labels, 'cids': np.arange(len(self.labels))})
        df['path'] = self.images_dir + df.path
        train_df = df[df.pids < 99]
        test_df = df[df.pids >= 99]
        self.trainval = train_df.to_records(index=False).tolist()
        self.query = self.gallery = test_df.to_records(index=False).tolist()

        self.val = None
        self.train = self.trainval
        self.num_val_ids = 0
        self.num_train_ids = self.num_trainval_ids = np.unique(train_df.pids).shape[0]
        self.num_query_ids = self.num_gallery_ids = np.unique(test_df.pids).shape[0]


if __name__ == '__main__':
    tic = time.time()
    # Market1501()
    # CUB()
    # Stanford_Prod()
    Car196()
    # iLIDSVID()
    # PRID()

    # Extract()
    # print(time.time() - tic)
    # import lmdb, lz
    #
    # ds = Mars()
    #
    # data_list = []
    # for dss in [ds.trainval, ds.query, ds.gallery]:
    #     for fns, pid, cid in dss:
    #         for fn in fns:
    #             # data_list.append(osp.basename(fn))
    #             data_list.append(fn)
    # data_list = np.asarray(data_list)
    # num_data = len(data_list)
    # max_map_size = int(num_data * 500 ** 2 * 3)  # be careful with this
    # env = lmdb.open('/home/xinglu/.torch/data/mars/img_lmdb', map_size=max_map_size)
    #
    # for img_path in data_list:
    #     with env.begin(write=True) as txn:
    #         with open(img_path, 'rb') as imgf:
    #             imgb = imgf.read()
    #         txn.put(osp.basename(img_path).encode(), imgb)

from fuel.datasets import H5PYDataset
from fuel.utils import find_in_data_path


class CUB2(Dataset):
    _filename = 'cub200_2011/cub200_2011.hdf5'

    def __init__(self, split='train', **kwargs):
        path = find_in_data_path(self._filename)
        self.split = split
        self.train = H5PYDataset(file_or_path=path, which_sets=['train'])
        self.train_labels = H5PYDataset(file_or_path=path,
                                        which_sets=['train'], sources=['targets'],
                                        load_in_memory=True).data_sources[0].ravel()
        self.test = H5PYDataset(file_or_path=path, which_sets=['test'])
        self.test_labels = H5PYDataset(file_or_path=path,
                                       which_sets=['test'], sources=['targets'],
                                       load_in_memory=True).data_sources[0].ravel()
        self.train_handle = self.train.open()
        self.test_hanle = self.test.open()
        self.ntest = self.test.num_examples
        self.ntrain = self.train.num_examples

    def _get_single_item(self, index):
        if self.split == 'train':
            img, label = self.train.get_data(self.train_handle, [index])
        else:
            img, label = self.test.get_data(self.test_hanle, [index])
        return img[0].transpose(1, 2, 0), label[0]

    def _get_multiple_items(self, idxs):
        if self.split == 'train':
            img, label = self.train.get_data(self.train_handle, idxs)
        else:
            img, label = self.test.get_data(self.test_hanle, idxs)
        return img.transpose(0, 2, 3, 1), label.ravel()

    def __getitem__(self, item):
        if isinstance(item, (tuple, list)):
            return self._get_multiple_items(item)
        else:
            return self._get_single_item(item)

    def __len__(self):
        if self.split == 'train':
            return self.ntrain
        else:
            return self.ntest

    def close(self):
        self.train_handle.close()
        self.test_hanle.close()

    def __del__(self):
        self.close()

    def __exit__(self):
        self.close()


class Stanford_prod2(Dataset):
    pass


from scipy.io import loadmat
