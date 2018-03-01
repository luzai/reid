from .cuhk01 import CUHK01
from .cuhk03 import CUHK03
from .dukemtmc import DukeMTMC
from .market1501 import Market1501
from reid.datasets.cdm import CDM
from .viper import VIPeR
from lz import *
from reid.utils.data import Dataset

__factory = {
    'viper': VIPeR,
    'cuhk01': CUHK01,
    'cuhk03': CUHK03,  # is 'cuhk03/label' default
    'cuhk03/label': CUHK03, 'cuhk03/detect': CUHK03, 'cuhk03/combine': CUHK03,
    'market1501': Market1501,
    'dukemtmc': DukeMTMC,
    'cdm':CDM
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def creates(names, roots, *args, **kwargs):
    dss = [create(name, root, *args, **kwargs) for name, root in zip(names, roots)]
    dsf = Dataset(root='', split_id=0)

    if osp.exists('/home/xinglu/work/cache.pkl'):
        dsf_dict = unpickle('/home/xinglu/work/cache.pkl')
        for k, v in dsf_dict.items():
            setattr(dsf, k, v)

        return dsf

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

# if __name__ == '__main__':
#     root = '/data1/xinglu/.torch/data/'
#     names = ['cuhk03', 'market1501', 'dukemtmc']
#     roots = [root + name  for name in names]
#     dataset = creates( names , roots)
#     print(dataset)