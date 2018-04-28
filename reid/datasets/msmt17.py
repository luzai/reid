from lz import *

from reid.utils.data import Dataset
from reid.utils.serialization import write_json
from reid.datasets import *


class MSMT17(Dataset):
    def __init__(self, root, split_id=0, num_val=100,
                 **kwargs):
        super(MSMT17, self).__init__(root, split_id=split_id)

        # self.build()
        self.load(num_val)



    def build(self):
        # todo
        for p in ['gallery', 'query', 'train', 'val']:
            pp = self.root + '/MSMT17_V1/list_' + p + '.txt'
            setattr(self,p, pd.read_csv(pp, delim_whitespace=True, header=None) )


if __name__ == '__main__':
    ds = MSMT17('/home/xinglu/.torch/data/msmt17')
