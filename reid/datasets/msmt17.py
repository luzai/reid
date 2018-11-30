from lz import *

from reid.utils.data import Dataset
from reid.utils.serialization import write_json
from reid.datasets import *


class MSMT17(Dataset):
    def __init__(self, root=None, split_id=0, num_val=100,
                 **kwargs):
        super(MSMT17, self).__init__(root, split_id=split_id)
        self.root = '/share/data/msmt17/'
        self.build()
        # self.load(num_val)

    def build(self):
        # self.load(100)
        # self.gallery query trainval train val
        # num_trainval_ids num_train_ids num_val_ids images_dir
        # fmt: (imagepath pids cids)
        self.images_dir = self.root + '/MSMT17_V1/'
        for p in ['gallery', 'query', 'train', 'val']:
            pp = self.root + '/MSMT17_V1/list_' + p + '.txt'
            tt = pd.read_csv(pp, delim_whitespace=True, header=None)
            tt.columns = ['images_dir', 'pid']
            tt.iloc[:, 1] = tt.iloc[:, 1].astype(int)
            pat = r'^(.*)'
            if p in ['gallery', 'query']:
                repl = lambda m: self.images_dir + 'test/' + m.group(0)
            else:
                repl = lambda m: self.images_dir + 'train/' + m.group(0)

            tt.iloc[:, 0] = tt.iloc[:, 0].str.replace(pat, repl)
            tt['cid'] = tt.iloc[:, 0].str.split('_', expand=True).iloc[:, 2].astype('int')
            if p == 'train':
                self._train_ids = np.unique(tt.pid)
                self.num_train_ids = np.unique(tt.pid).shape[0]
            elif p == 'val':
                self._val_ids = np.unique(tt.pid)
                self.num_val_ids = np.unique(tt.pid).shape[0]
            elif p == 'query':
                self._query_ids = np.unique(tt.pid)
            elif p == 'gallery':
                self._gallery_ids = np.unique(tt.pid)

            tt = tt.to_records(index=False).tolist()
            setattr(self, p, tt)
        self.trainval = self.train + self.val
        self.num_trainval_ids = np.unique(np.concatenate((self._train_ids, self._val_ids), )).shape[0]
        # self.query = self.query[:10]
        # self.gallery = self.gallery[:10]
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
              .format(self._query_ids.shape[0], len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self._gallery_ids.shape[0], len(self.gallery)))


if __name__ == '__main__':
    ds = MSMT17('//share/data/msmt17')
