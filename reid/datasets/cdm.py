import os.path as osp

from reid.utils.data import Dataset
from reid.utils.serialization import write_json
from reid.datasets import *

from lz import *


def creates(names, roots, *args, **kwargs):
    # if osp.exists('/home/xinglu/work/cache.pkl'):
    #     dsf_dict = unpickle('/home/xinglu/work/cache.pkl')
    #     for k, v in dsf_dict.items():
    #         setattr(dsf, k, v)
    #     return dsf
    dss = [create(name, root, *args, **kwargs) for name, root in zip(names, roots)]
    dsf = Dataset(root='', split_id=0)

    def to_df(rec):
        return pd.DataFrame.from_records(rec, columns=['fname', 'pid', 'cid'])

    def combine(name):
        df_l = []
        #     now_pid=0
        for ds in dss:
            df = to_df(getattr(ds, name))
            df['path'] = ds.root + '/images'
            df['name'] = osp.basename(ds.root)
            df.pid = df.name + '_' + df.pid.astype(str)
            #         df.pid+=now_pid
            #         now_pid=df.pid.max()+1
            #         print(now_pid)
            df_l.append(df)

        df_f = pd.concat(df_l)
        return df_f

    for name in ['trainval', 'val', 'train', 'query', 'gallery']:
        dff = combine(name)
        dff.fname = dff.path + '/' + dff.fname
        setattr(dsf, name, dff)

    dft = pd.concat([dsf.trainval, dsf.query, dsf.gallery])
    pids = np.unique(dft.pid)
    pids = np.sort(pids)
    # pids.shape

    mapp = dict(zip(pids, np.arange(pids.shape[0])))

    for name in ['trainval', 'val', 'train', 'query', 'gallery']:
        getattr(dsf, name).pid.replace(mapp, inplace=True)
        getattr(dsf, name).reset_index(inplace=True, drop=True)

    dsf.train.reset_index(inplace=True, drop=True)

    dsf.num_train_ids = np.unique(dsf.train.pid).shape[0]
    dsf.num_val_ids = np.unique(dsf.val.pid).shape[0]
    dsf.num_trainval_ids = np.unique(dsf.trainval.pid).shape[0]

    dsf.split = {'query': np.unique(dsf.query.pid.values.tolist()),
                 'gallery': np.unique(dsf.gallery.pid.values).tolist()}

    for name in ['trainval', 'val', 'train', 'query', 'gallery']:
        dff = getattr(dsf, name)
        del dff['path'], dff['name']
        dff = dff.to_records(index=False)
        dff = dff.tolist()
        setattr(dsf, name, dff)

    return dsf


class CDM(Dataset):
    def __init__(self, root, split_id=0, num_val=100,
                 **kwargs):
        super(CDM, self).__init__(root, split_id=split_id)

        # ds = self.combine()
        # print(len(ds.trainval))
        self.load(num_val)

    def combine(self):
        names = ['cuhk03', 'market1501', 'dukemtmc']
        root = '/home/xinglu/.torch/data/'
        roots = [root + name_ for name_ in names]
        return creates(names, roots)


if __name__ == '__main__':
    ds = CDM('/home/xinglu/.torch/data/cdm')
    len(ds.trainval)
