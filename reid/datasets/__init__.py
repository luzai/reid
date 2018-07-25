from reid.datasets.cuhk01 import CUHK01
from reid.datasets.cuhk03 import CUHK03
from reid.datasets.dukemtmc import DukeMTMC
from reid.datasets.market1501 import *
from reid.datasets.msmt17 import MSMT17
from reid.datasets.cdm import CDM
from reid.datasets.viper import VIPeR
from reid.utils.data import Dataset
import lz

cls2name = {
    VIPeR: ['viper'],
    CUHK03: ['cu03det', 'cu03lbl', 'cu03det.classic', 'cu03lbl.classic', 'cuhk03'],
    Market1501: ['mkt', 'market1501'],
    DukeMTMC: ['dukemtmc', ],
    CDM: ['cdm'],
    MSMT17: ['msmt17'],
    Mars: ['mars'],
    iLIDSVID: ['ilidsvid'],
    PRID: ['prid'],
    CUB: ['cub'],
    Stanford_Prod: ['stanford_prod']
}

__factory = {v1: i for i, v in cls2name.items() for v1 in v}


def names():
    return sorted(__factory.keys())


def parse_name(ds):
    args_ds = edict()
    if ds == 'cu03det':
        args_ds.dataset = 'cuhk03'
        args_ds.dataset_val = 'cuhk03'
        args_ds.dataset_mode = 'detect'
        args_ds.eval_conf = 'cuhk03'
    elif ds == 'cu03lbl':
        args_ds.dataset = 'cuhk03'
        args_ds.dataset_val = 'cuhk03'
        args_ds.dataset_mode = 'label'
        args_ds.eval_conf = 'cuhk03'
    elif ds == 'mkt' or ds == 'market' or ds == 'market1501':
        args_ds.dataset = 'market1501'
        args_ds.dataset_val = 'market1501'
        args_ds.eval_conf = 'market1501'
    elif ds == 'msmt':
        args_ds.dataset = 'msmt17'
        args_ds.dataset_val = 'market1501'
        args_ds.eval_conf = 'market1501'
    elif ds == 'cdm':
        args_ds.dataset = 'cdm'
        args_ds.dataset_val = 'market1501'
        args_ds.eval_conf = 'market1501'
    elif ds == 'viper':
        args_ds.dataset = 'viper'
        args_ds.dataset_val = 'viper'
        args_ds.eval_conf = 'market1501'
    elif ds == 'cu01hard':
        args_ds.dataset = 'cuhk01'
        args_ds.dataset_val = 'cuhk01'
        args_ds.eval_conf = 'cuhk03'
        args_ds.dataset_mode = 'hard'
    elif ds == 'cu01easy':
        args_ds.dataset = 'cuhk01'
        args_ds.dataset_val = 'cuhk01'
        args_ds.eval_conf = 'cuhk03'
        args_ds.dataset_mode = 'easy'
    elif ds == 'dukemtmc':
        args_ds.dataset = 'dukemtmc'
        args_ds.dataset_val = 'dukemtmc'
        args_ds.eval_conf = 'market1501'
    else:
        # raise ValueError(f'dataset ... {ds}')
        args_ds.dataset_val = args_ds.dataset
        args_ds.eval_conf = 'market1501'
    return args_ds


def create(name, *args, **kwargs):
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
    args_ds = parse_name(name)
    kwargs = lz.dict_update(kwargs, args_ds, must_exist=False)
    root = '/home/xinglu/.torch/data/' + kwargs.get('dataset')
    return __factory[kwargs.get('dataset')](root=root, *args, **kwargs)

# if __name__ == '__main__':
#     root = '/data1/xinglu/.torchet/data/'
#     names = ['cuhk03', 'market1501', 'dukemtmc']
#     roots = [root + name  for name in names]
#     dataset = creates( names , roots)
#     print(dataset)
