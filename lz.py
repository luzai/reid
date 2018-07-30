# # -*- coding: future_fstrings -*-
# from __future__ import print_function, absolute_import, division, unicode_literals

# import matplotlib
# # matplotlib.use('TkAgg')
# # matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import os, sys, time, \
    random, \
    subprocess, glob, re, \
    numpy as np, \
    multiprocessing as mp, \
    logging, \
    collections, \
    functools, signal
from os import path as osp

# from easydict import EasyDict as edict
# import cv2, cvbase as cvb, copy ,math
# import redis, networkx as nx, \
#  yaml, subprocess, pprint, json, \
# csv, argparse, string, colorlog, \
# shutil, itertools, pandas as pd, pathlib,
# from IPython import embed
# from tensorboardX import SummaryWriter


if os.environ.get('pytorch', "1") == "1":
    tic = time.time()
    import torch
    import torchvision
    import torch.utils.data
    from torch import nn
    import torch.nn.functional as F

    old_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = lambda obj: (f'{tuple(obj.shape)} {obj.type()} \n'
                                         f'{old_repr(obj)} \n'
                                         f'type: {obj.type()} shape: {obj.shape}') if obj.is_contiguous() else (
        f'{tuple(obj.shape)} {obj.type()} \n'
        f'{old_repr(obj.contiguous())} \n'
        f'type: {obj.type()} shape: {obj.shape}')
    print('import pytorch', time.time() - tic)
else:
    tic = time.time()
    import tensorflow as tf
    import tensorflow.contrib
    import tensorflow.contrib.keras


    def allow_growth():
        import tensorflow as tf
        oldinit = tf.Session.__init__

        def myinit(session_object, target='', graph=None, config=None):
            if config is None:
                config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            oldinit(session_object, target, graph, config)

        tf.Session.__init__ = myinit
        return oldinit


    oldinit = allow_growth()
    print('import tf', time.time() - tic)

root_path = osp.normpath(
    osp.join(osp.abspath(osp.dirname(__file__)), )
) + '/'

home_path = os.environ['HOME'] + '/'
work_path = home_path + '/work/'
share_path = '/data1/share/'

'''
%load_ext autoreload
%autoreload 2
%matplotlib inline
import matplotlib
matplotlib.style.use('ggplot')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
'''
# torch.set_default_tensor_type(torch.cuda.DoubleTensor)
ori_np_err = np.seterr(all='raise')


def set_stream_logger(log_level=logging.DEBUG):
    import colorlog
    sh = colorlog.StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(
        colorlog.ColoredFormatter(
            ' %(asctime)s %(filename)s [line:%(lineno)d] %(log_color)s%(levelname)s%(reset)s %(message)s'))
    logging.root.addHandler(sh)


def set_file_logger(work_dir=None, log_level=logging.DEBUG):
    work_dir = work_dir or root_path
    fh = logging.FileHandler(os.path.join(work_dir, 'log-ing'))
    fh.setLevel(log_level)
    fh.setFormatter(
        logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s'))
    logging.root.addHandler(fh)


logging.root.setLevel(logging.DEBUG)
# set_stream_logger(logging.DEBUG)
set_stream_logger(logging.INFO)
set_file_logger(log_level=logging.DEBUG)

## ndarray will be pretty
# np.set_string_function(lambda arr: f'{arr.shape} {arr.dtype} \n'
#                                    f'{arr.__str__()} \n'
#                                    f'dtype:{arr.dtype} shape:{arr.shape}'                       , repr=True)

## print(ndarray) will be pretty (and pycharm dbg)
np.set_string_function(lambda arr: f'{arr.shape} {arr.dtype} \n'
                                   f'{arr.__repr__()} \n'
                                   f'dtype:{arr.dtype} shape:{arr.shape}', repr=False)


# old_np_repr = np.ndarray.__repr__
# np.ndarray.__repr__ = lambda arr: (f'{arr.shape} {arr.dtype} \n'
#                                    f'{old_np_repr(arr)} \n'
#                                    f'dtype:{arr.dtype} shape:{arr.shape}')

def init_dev(n=(0,)):
    import os
    import logging
    if not isinstance(n, collections.Sequence):
        n = (n,)
    logging.info('use gpu {}'.format(n))
    home = os.environ['HOME']
    if isinstance(n, int) or n is None:
        n = (n,)
    devs = ''
    for n_ in n:
        devs += str(n_) + ','
    os.environ["CUDA_VISIBLE_DEVICES"] = devs
    set_env('PATH', home + '/local/cuda/bin')
    set_env('LD_LIBRARY_PATH', home + '/local/cuda/lib64:' +
            home + '/local/cuda/extras/CUPTI/lib64')


def set_env(key, value):
    if key in os.environ:
        os.environ[key] = value + ':' + os.environ[key]
    else:
        os.environ[key] = value


def occupy(dev=range(8)):
    import tensorflow as tf
    init_dev(dev)
    newinit = tf.Session.__init__
    if 'oldinit' in locals():
        tf.Session.__init__ = oldinit
    var = tf.constant(1)
    with tf.Session() as sess:
        sess.run([var])
    while True:
        time.sleep(10)
    # tf.Session.__init__ = newinit


# if something like Runtime Error : an illegal memory access was encountered occur
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


'''
oldinit = Session.__init__

def myinit(session_object, target='', graph=None, config=None):
    if config is None:
        config = ConfigProto()
    config.gpu_options.allow_growth = True
    oldinit(session_object, target, graph, config)

Session.__init__ = myinit
'''


def allow_growth_conf():
    import tensorflow as tf
    _sess_config = tf.ConfigProto(allow_soft_placement=True)
    _sess_config.gpu_options.allow_growth = True
    return _sess_config


def allow_growth_sess():
    import tensorflow as tf
    tf_graph = tf.get_default_graph()
    _sess_config = tf.ConfigProto(allow_soft_placement=True)
    _sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=_sess_config, graph=tf_graph)
    return sess


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def get_mem(ind=0):
    import gpustat
    gpus = gpustat.GPUStatCollection.new_query().gpus
    return gpus[ind].entry['memory.used'] / gpus[ind].entry['memory.total'] * 100


def get_utility(ind=0):
    import gpustat
    gpus = gpustat.GPUStatCollection.new_query().gpus
    return gpus[ind].entry['utilization.gpu']


def show_dev(devs=range(4)):
    res = []
    for ind in devs:
        mem = get_mem(ind)
        print(ind, mem)
        res.append(mem)
    return res


def get_dev(n=1, ok=range(4), mem_thresh=(0.1, 0.15), sleep=20):
    if not isinstance(mem_thresh, collections.Sequence):
        mem_thresh = (mem_thresh,)

    def get_poss_dev():
        mems = [get_mem(ind) for ind in ok]
        inds, mems = cosort(ok, mems, return_val=True)
        devs = [ind for ind, mem in zip(inds, mems) if mem < mem_thresh[0] * 100]

        return devs

    devs = get_poss_dev()
    logging.info('Auto select gpu')
    # gpustat.print_gpustat()
    show_dev()
    while len(devs) < n:
        devs = get_poss_dev()

        print('no enough device available')
        # gpustat.print_gpustat()
        show_dev()

        sleep = int(sleep)
        time.sleep(random.randint(max(0, sleep - 20), sleep + 20))
    return devs[:n]


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


def cpu_priority(level=19):
    import psutil
    p = psutil.Process(os.getpid())
    p.nice(level)


def mkdir_p(path, delete=True):
    if path == '':
        return
    if delete and osp.exists(path):
        rm(path)
    if not osp.exists(path):
        shell('mkdir -p ' + path)


class Logger(object):
    def __init__(self, fpath=None, console=sys.stdout):
        self.console = console
        self.file = None
        if fpath is not None:
            mkdir_p(os.path.dirname(fpath), delete=False)
            # rm(fpath)
            self.file = open(fpath, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


sys.stdout = Logger(root_path + 'log-prt')
sys.stderr = Logger(root_path + 'log-prt-err')


class Timer(object):
    """A flexible Timer class.

    :Example:

    >>> import time
    >>> import cvbase as cvb
    >>> with cvb.Timer():
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    1.000
    >>> with cvb.Timer(print_tmpl='hey it taks {:.1f} seconds'):
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    hey it taks 1.0 seconds
    >>> timer = cvb.Timer()
    >>> time.sleep(0.5)
    >>> print(timer.since_start())
    0.500
    >>> time.sleep(0.5)
    >>> print(timer.since_last_check())
    0.500
    >>> print(timer.since_start())
    1.000

    """

    def __init__(self, start=True, print_tmpl=None):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else '{:.3f}'
        if start:
            self.start()

    @property
    def is_running(self):
        """bool: indicate whether the timer is running"""
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def start(self):
        """Start the timer."""
        if not self._is_running:
            self._t_start = time.time()
            self._is_running = True
        self._t_last = time.time()

    def since_start(self, aux=''):
        """Total time since the timer is started.

        Returns(float): the time in seconds
        """
        if not self._is_running:
            raise ValueError('timer is not running')
        self._t_last = time.time()
        logging.info(f'{aux} time {self.print_tmpl.format(self._t_last - self._t_start)}')
        return self._t_last - self._t_start

    def since_last_check(self, aux=''):
        """Time since the last checking.

        Either :func:`since_start` or :func:`since_last_check` is a checking operation.

        Returns(float): the time in seconds
        """
        if not self._is_running:
            raise ValueError('timer is not running')
        dur = time.time() - self._t_last
        self._t_last = time.time()
        logging.info(f'{aux} time {self.print_tmpl.format(dur)}')
        return dur


def get_md5(url):
    if isinstance(url, str):
        url = url.encode('utf-8')
    import hashlib
    m = hashlib.md5()
    m.update(url)
    return m.hexdigest()


def load_cfg(cfg_file):
    from importlib import import_module
    sys.path.append(osp.dirname(cfg_file))
    module_name = osp.basename(cfg_file).rstrip('.py')
    cfg = import_module(module_name)
    return cfg


# Based on an original idea by https://gist.github.com/nonZero/2907502 and heavily modified.
class Uninterrupt(object):
    """
    Use as:
    with Uninterrupt() as u:
        while not u.interrupted:
            # train
    """

    def __init__(self, sigs=(signal.SIGINT,), verbose=False):
        self.sigs = sigs
        self.verbose = verbose
        self.interrupted = False
        self.orig_handlers = None

    def __enter__(self):
        if self.orig_handlers is not None:
            raise ValueError("Can only enter `Uninterrupt` once!")

        self.interrupted = False
        self.orig_handlers = [signal.getsignal(sig) for sig in self.sigs]

        def handler(signum, frame):
            self.release()
            self.interrupted = True
            if self.verbose:
                print("Interruption scheduled...", flush=True)

        for sig in self.sigs:
            signal.signal(sig, handler)

        return self

    def __exit__(self, type_, value, tb):
        self.release()

    def release(self):
        if self.orig_handlers is not None:
            for sig, orig in zip(self.sigs, self.orig_handlers):
                signal.signal(sig, orig)
            self.orig_handlers = None


def mail(content, to_mail=('907682447@qq.com',)):
    import datetime, collections
    user_pass = {'username': '907682447@qq.com',
                 'password': 'luzai123',
                 'host': 'smtp.qq.com',
                 'port': 587}

    # user_pass = {'username': 'wxlms@outlook.com',
    #              'password': 'yana3140102282',
    #              'host': 'smtp.outlook.com',
    #              'port': 587}
    # user_pass = {'username': '3140102282@zju.edu.cn',
    #              'password': 'eePh9zie',
    #              'host': 'smtp.zju.edu.cn',
    #              'port': 25}

    time_str = datetime.datetime.now().strftime('%m-%d %H:%M')

    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    s = smtplib.SMTP(host=user_pass['host'], port=user_pass['port'], timeout=10)
    s.starttls()
    s.login(user_pass['username'], user_pass['password'])

    title = 'ps: ' + content.split('\r\n')[0]
    title = title[:20]
    content = time_str + '\r\n' + content
    if isinstance(to_mail, collections.Sequence):
        to_mail = ', '.join(to_mail)
    msg = MIMEMultipart('alternative')
    msg['Subject'] = title
    msg['From'] = user_pass['username']
    msg['To'] = to_mail
    # msg['Cc'] = to_mail
    msg.attach(MIMEText(content, 'plain'))
    s.sendmail(msg['From'], msg['To'], msg.as_string())
    s.quit()


def df2md(df1):
    import tabulate
    return tabulate.tabulate(df1, headers="keys", tablefmt="pipe")


def stat_np(array):
    return np.min(array), np.mean(array), np.median(array), np.max(array)


def stat_th(tensor):
    return torch.min(tensor), torch.mean(tensor), torch.median(tensor), torch.max(tensor)


def sel_np(A):
    import json
    dtype = str(A.dtype)
    shape = A.shape
    A = A.ravel().tolist()
    sav = {'shape': shape, 'dtype': dtype,
           'A': A
           }
    return json.dumps(sav)


def desel_np(s):
    import json
    sav = json.loads(s)
    A = sav['A']
    A = np.array(A, dtype=sav['dtype']).reshape(sav['shape'])
    return A


def to_numpy(tensor):
    if isinstance(tensor, torch.autograd.Variable):
        tensor = tensor.detach()
    if torch.is_tensor(tensor):
        if tensor.shape == ():
            tensor = tensor.item()
            tensor = np.asarray([tensor])
        elif np.prod(tensor.shape) == 1:
            tensor = tensor.item()
            tensor = np.asarray([tensor])
        else:
            tensor = tensor.cpu().numpy()
            tensor = np.asarray(tensor)
    # elif type(tensor).__module__ != 'numpy':
    #     raise ValueError("Cannot convert {} to numpy array"
    #                      .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if ndarray is None:
        return None
    if isinstance(ndarray, collections.Sequence):
        return [to_torch(ndarray_) for ndarray_ in ndarray if ndarray_ is not None]
    # if isinstance(ndarray, torch.autograd.Variable):
    #     ndarray = ndarray.data
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def to_variable(tn, volatile=False, requires_grad=False, cuda=False, **kwargs):
    if tn is None:
        return None
    if isinstance(tn, collections.Sequence):
        return [to_variable(tn_, **kwargs) for tn_ in tn if tn_ is not None]
    tn = to_torch(tn)
    if cuda and torch.cuda.is_available():
        tn = tn.cuda()
    if not volatile or requires_grad:
        tn.requires_grad = True
        return tn
    else:
        with torch.no_grad():
            return tn


def norm_np(tensor):
    min, max = tensor.min(), tensor.max()
    tensor += min
    tensor /= (max - min)
    tensor *= 255
    return tensor


def norm_th(tensor):
    min, max = tensor.min(), tensor.max()
    return tensor.add_(min).div_(max - min)


def load_state_dict(model, state_dict, own_prefix='', own_de_prefix=''):
    own_state = model.state_dict()
    success = []
    for name, param in state_dict.items():
        if own_prefix + name in own_state:
            name = own_prefix + name
        if name.replace(own_de_prefix, '') in own_state:
            name = name.replace(own_de_prefix, '')

        if name not in own_state:
            print('ignore key "{}" in his state_dict'.format(name))
            continue

        if isinstance(param, nn.Parameter):
            param = param.clone()

        if own_state[name].size() == param.size():
            own_state[name].copy_(param)
            # print('{} {} is ok '.format(name, param.size()))
            success.append(name)
        else:
            logging.error('dimension mismatch for param "{}", in the model are {}'
                          ' and in the checkpoint are {}, ...'.format(
                name, own_state[name].size(), param.size()))

    missing = set(own_state.keys()) - set(success)
    if len(missing) > 0:
        print('missing keys in my state_dict: "{}"'.format(missing))


def grid_iter(*args):
    import itertools
    res = list(itertools.product(*args))
    np.random.shuffle(res)
    for arg in res:
        if len(arg) == 1:
            yield arg[0]
        else:
            yield arg


def cross_iter(*args):
    start = [t[0] for t in args]
    yield start
    for ind, arg in enumerate(args):
        if len(arg) > 1:
            bak = start[ind]
            for ar in arg[1:]:
                start[ind] = ar
                yield start
            start[ind] = bak


def shuffle_iter(iter):
    iter = list(iter)
    np.random.shuffle(iter)
    for iter_ in iter:
        yield iter_


def optional_arg_decorator(fn):
    def wrapped_decorator(*args):
        if len(args) == 1 and callable(args[0]):
            return fn(args[0])
        else:
            def real_decorator(decoratee):
                return fn(decoratee, *args)

            return real_decorator

    return wrapped_decorator


def randomword(length):
    import random
    import string
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def cosort(ind, val, return_val=False):
    ind = np.asarray(ind)
    val = np.asarray(val)
    comb = zip(ind, val)
    comb_sorted = sorted(comb, key=lambda x: x[1])
    if not return_val:
        return np.array([comb_[0] for comb_ in comb_sorted])
    else:
        return np.array([comb_[0] for comb_ in comb_sorted]), np.array([comb_[1] for comb_ in
                                                                        comb_sorted])


@optional_arg_decorator
def timeit(fn, info=''):
    def wrapped_fn(*arg, **kwargs):
        start = time.time()
        res = fn(*arg, **kwargs)
        diff = time.time() - start
        logging.info((info + 'takes time {}').format(diff))
        return res

    return wrapped_fn


class Database(object):
    def __init__(self, file, mode='a'):
        import h5py
        try:
            self.fid = h5py.File(file, mode)
        except OSError as inst:
            logging.error(f'{inst}')
            rm(file)
            self.fid = h5py.File(file, 'w')
            logging.error(f'{file} is delete and write !!')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fid.close()

    def __getitem__(self, keys):
        if isinstance(keys, (tuple, list)):
            return [self._get_single_item(k) for k in keys]
        return self._get_single_item(keys)

    def _get_single_item(self, key):
        return np.asarray(self.fid[key])

    def __setitem__(self, key, value):
        value = np.asarray(value)
        if key in self.fid:
            if self.fid[key].shape == value.shape and \
                    self.fid[key].dtype == value.dtype:
                logging.debug('shape type same, old is updated, {} {} '.format(value, np.count_nonzero(value == -1))
                              )
                self.fid[key][...] = value
            else:
                logging.debug('old shape {} new shape {} updated'.format(
                    self.fid[key].shape, value.shape))
                del self.fid[key]
                self.fid.create_dataset(key, data=value)
        else:
            self.fid.create_dataset(key, data=value)

    def __delitem__(self, key):
        del self.fid[key]

    def __len__(self):
        return len(self.fid)

    def __iter__(self):
        return iter(self.fid)

    def flush(self):
        self.fid.flush()

    def close(self):
        self.flush()
        self.fid.close()

    def keys(self):
        return self.fid.keys()


def pickle_dump(data, file, **kwargs):
    import pickle, pathlib
    # python2 can read 2
    kwargs.setdefault('protocol', pickle.HIGHEST_PROTOCOL)
    if isinstance(file, str) or isinstance(file, pathlib.Path):
        mkdir_p(osp.dirname(file), delete=False)
        print('pickle into', file)
        with open(file, 'wb') as f:
            pickle.dump(data, f, **kwargs)
    elif hasattr(file, 'write'):
        pickle.dump(data, file, **kwargs)
    else:
        raise TypeError("file must be str of file-object")


def get_img_size(img='/data1/xinglu/prj/test.jpg', verbose=True):
    out, err = shell(f'convert "{img}" -print "(%w, %h)" ', verbose=verbose)
    out = eval(out)
    out = (out[1], out[0])
    return out


def pickle_load(file, **kwargs):
    import pickle
    if isinstance(file, str):
        with open(file, 'rb') as f:
            data = pickle.load(f, **kwargs)
    elif hasattr(file, 'read'):
        data = pickle.load(file, **kwargs)
    return data


def dataframe_dump(df, path):
    df.to_hdf(path, 'df', mode='w')


def dataframe_load(path):
    import pandas as pd
    return pd.read_hdf(path, 'df')


def yaml_load(file, **kwargs):
    from yaml import Loader
    import yaml
    kwargs.setdefault('Loader', Loader)
    if isinstance(file, str):
        with open(file, 'r') as f:
            obj = yaml.load(f, **kwargs)
    elif hasattr(file, 'read'):
        obj = yaml.load(file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
    return obj


def yaml_dump(obj, file=None, **kwargs):
    import yaml
    from yaml import Dumper
    kwargs.setdefault('Dumper', Dumper)
    if file is None:
        return yaml.dump(obj, **kwargs)
    elif isinstance(file, str):
        with open(file, 'w') as f:
            yaml.dump(obj, f, **kwargs)
    elif hasattr(file, 'write'):
        yaml.dump(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def json_dump(obj, file, mode='a'):  # write not append!
    # import codecs
    import json
    if isinstance(file, str):
        # with codecs.open(file, mode, encoding='utf-8') as fp:
        with open(file, 'w') as fp:
            json.dump(obj, fp,
                      # ensure_ascii=False
                      )
    elif hasattr(file, 'write'):
        json.dump(obj, file)


def msgpack_dump(obj, file, **kwargs):
    import msgpack, msgpack_numpy as m
    kwargs.setdefault('allow_np', True)
    allow_np = kwargs.pop('allow_np')
    if allow_np:
        kwargs.setdefault('default', m.encode)
    kwargs.setdefault('use_bin_type', True)

    with open(file, 'wb') as fp:
        msgpack.pack(obj, fp, **kwargs)


def msgpack_dumps(obj, **kwargs):
    import msgpack, msgpack_numpy as m
    kwargs.setdefault('allow_np', True)
    allow_np = kwargs.pop('allow_np')
    if allow_np:
        kwargs.setdefault('default', m.encode)
    kwargs.setdefault('use_bin_type', True)

    return msgpack.packb(obj, **kwargs)


def msgpack_load(file, **kwargs):
    import msgpack, gc, msgpack_numpy as m
    gc.disable()
    kwargs.setdefault('allow_np', True)
    allow_np = kwargs.pop('allow_np')
    if allow_np:
        kwargs.setdefault('object_hook', m.decode)
    kwargs.setdefault('use_list', False)
    kwargs.setdefault('raw', False)
    with open(file, 'rb') as f:
        res = msgpack.unpack(f, **kwargs)
    gc.enable()
    return res


def msgpack_loads(file, **kwargs):
    pass


# import msgpack_numpy as m
#
# msgpack_dump(np.zeros((2, 2)), '/tmp/tmp', default=m.encode)
# msgpack_load('/tmp/tmp', object_hook=m.decode)


def json_load(file):
    import json
    if isinstance(file, str):
        with open(file, 'r') as f:
            obj = json.load(f)
    elif hasattr(file, 'read'):
        obj = json.load(file)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
    return obj


def append_file(line, file=None):
    file = file or 'append.txt'
    with open(file, 'a') as f:
        f.writelines(line + '\n')


def write_list(file, l, sort=True, delimiter=' ', fmt='%.18e'):
    l = np.array(l)
    if sort:
        l = np.sort(l, axis=0)
    np.savetxt(file, l, delimiter=delimiter, fmt=fmt)


class AsyncDumper(mp.Process):
    def __init__(self):
        self.queue = mp.Queue()
        super(AsyncDumper, self).__init__()

    def run(self):
        while True:
            data, out_file = self.queue.get()
            if data is None:
                break
            pickle_dump(data, out_file)

    def dump(self, obj, filename):
        self.queue.put((obj, filename))


def shell(cmd, block=True, return_msg=True, verbose=True, timeout=None):
    import os
    my_env = os.environ.copy()
    home = os.path.expanduser('~')
    my_env['PATH'] = home + "/anaconda3/bin/:" + my_env['PATH']
    if verbose:
        logging.info('cmd is ' + cmd)
    if block:
        # subprocess.call(cmd.split())
        task = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=my_env,
                                preexec_fn=os.setsid
                                )
        if return_msg:
            msg = task.communicate(timeout)
            msg = [msg_.decode('utf-8') for msg_ in msg]
            if msg[0] != '' and verbose:
                logging.info('stdout {}'.format(msg[0]))
            if msg[1] != '' and verbose:
                logging.error('stderr {}'.format(msg[1]))
            return msg
        else:
            return task
    else:
        print('Non-block!')
        task = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=my_env,
                                preexec_fn=os.setsid
                                )
        return task


def ln(path, to_path):
    if osp.exists(to_path):
        print('error! exist ' + to_path)
    path = osp.abspath(path)
    cmd = "ln -s " + path + " " + to_path
    print(cmd)
    proc = subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    return proc


def tar(path, to_path=None):
    if not osp.exists(path):
        return
    if not osp.exists(to_path):
        mkdir_p(to_path)
    if os.path.exists(to_path) and not len(os.listdir(to_path)) == 0:
        rm(path)
        return
    if to_path is not None:
        cmd = "tar xf " + path + " -C " + to_path
        print(cmd)
    else:
        cmd = "tar xf " + path
    shell(cmd, block=True)
    if os.path.exists(path):
        rm(path)


def rm(path, block=True):
    path = osp.abspath(path)
    if not osp.exists(path):
        logging.info(f'no need rm {path}')
    stdout, _ = shell('which trash', verbose=False)
    if 'trash' not in stdout:
        dst = glob.glob('{}.bak*'.format(path))
        parsr = re.compile(r'{}.bak(\d+?)'.format(path))
        used = [0, ]
        for d in dst:
            m = re.match(parsr, d)
            if not m:
                used.append(0)
            elif m.groups()[0] == '':
                used.append(0)
            else:
                used.append(int(m.groups()[0]))
        dst_path = '{}.bak{}'.format(path, max(used) + 1)
        cmd = 'mv {} {} '.format(path, dst_path)
        return shell(cmd, block=block)
    else:
        return shell(f'trash -r {path}', block=block)


def show_img(path):
    from IPython.display import Image

    fig = Image(filename=path)
    return fig


def show_pdf(path):
    from IPython.display import IFrame
    path = osp.relpath(path)
    return IFrame(path, width=600, height=300)


def print_graph_info():
    import tensorflow as tf
    graph = tf.get_default_graph()
    graph.get_tensor_by_name("Placeholder:0")
    layers = [op.name for op in graph.get_operations() if op.type ==
              "Placeholder"]
    print([graph.get_tensor_by_name(layer + ":0") for layer in layers])
    print([op.type for op in graph.get_operations()])
    print([n.name for n in tf.get_default_graph().as_graph_def().node])
    print([v.name for v in tf.global_variables()])
    print(graph.get_operations()[20])


def chdir_to_root(fn):
    def wrapped_fn(*args, **kwargs):
        restore_path = os.getcwd()
        os.chdir(root_path)
        res = fn(*args, **kwargs)
        os.chdir(restore_path)
        return res

    return wrapped_fn


def scp(src, dest, dry_run=False):
    cmd = ('scp -r ' + src + ' ' + dest)
    print(cmd)
    if dry_run:
        return
    return shell(cmd, block=False)


def read_list(file, delimi=" "):
    if osp.exists(file):
        lines = np.genfromtxt(file, dtype='str', delimiter=delimi)
        return lines
    else:
        return []


def cp(from_path, to):
    dst_dir = osp.dirname(to)
    if not osp.exists(dst_dir):
        mkdir_p(dst_dir)
    shell('cp -r ' + from_path + ' ' + to)


def mv(from_path, to):
    if isinstance(from_path, list):
        for from_ in from_path:
            mv(from_, to)
    elif isinstance(to, list):
        for to_ in to:
            mv(from_path, to_)
    else:
        shell(f'''mv  "{from_path}" "{to}"''')


def dict_concat(d_l):
    d1 = d_l[0].copy()
    for d in d_l[1:]:
        d1.update(d)
    return d1


def dict_update(to, from_dict, must_exist=True):
    to = to.copy()
    from_dict = from_dict.copy()
    for k, v in from_dict.items():
        if k not in to:
            if not must_exist:
                logging.debug('ori dict do not have key {}'.format(k))
            else:
                raise ValueError('ori dict do not have key {}'.format(k))
        try:
            assert to[k] == v
        except Exception as inst:
            logging.debug(
                'update ori key {} from {} to {}'.format(k, to.get(k, None), v))
            to[k] = v
    return to


def face_detect(path='/data1/xinglu/prj/test.jpg'):
    cmd = f'''
curl -X POST "https://api-cn.faceplusplus.com/facepp/v3/detect" \
-F "api_key=cmWxHgmGAIglR3iWeJSZLioxkQop4EqW" \
-F "api_secret=nluB3GvYghHG4f5qb106uKwZbpGzxq94" \
-F "image_file=@{path}" \
-F "return_landmark=1" \
-F "return_attributes=gender,age,headpose,facequality"
    '''
    out, err = shell(cmd)
    return out


def clean_name(name):
    if isinstance(name, list):
        return [clean_name(n) for n in name]
    import re
    name = re.findall('([a-zA-Z0-9/-]+)(?::\d+)?', name)[0]
    name = re.findall('([a-zA-Z0-9/-]+)(?:_\d+)?', name)[0]
    return name


class Struct(object):
    def __init__(self, entries):
        self.__dict__.update(entries)

    def __getitem__(self, item):
        return self.__dict__[item]


def dict2obj(d):
    return Struct(d)


def dict2str(others):
    name = ''
    for key, val in others.iteritems():
        name += '_' + str(key)
        if isinstance(val, dict):
            name += '_' + dict2str(val)
        elif isinstance(val, list):
            for val_ in val:
                name += '-' + str(val_)
        else:
            name += '_' + str(val)
    return name


def list2str(li, delimier=''):
    name = ''
    for name_ in li:
        name += (str(name_) + delimier)

    return name


def rsync(from_, to):
    cmd = ('rsync -avzP ' + from_ + ' ' + to)
    print(cmd)
    return shell(cmd, block=False)


def i_vis_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    import tensorflow as tf
    from IPython.display import display, HTML, SVG
    import os

    def strip_consts(graph_def, max_const_size=32):
        """Strip large constant values from graph_def."""
        import tensorflow as tf

        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add()
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = tf.compat.as_bytes(
                        "<stripped %d bytes>" % size)
        return strip_def

    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


def my_wget(fid, fname):
    shell('rm -rf /tmp/cookies.txt')
    task = shell(
        f"wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={fid}' -O- ",
        return_msg=False
    )
    out, err = task.communicate()
    out = out.decode('utf-8')
    print(out)
    if len(re.findall(r'.*confirm=([0-9a-zA-Z_]+).*', out)) == 0:
        print('no confirm continue')
        return 100
    confirm = re.findall(r'.*confirm=([0-9a-zA-Z_]+).*', out)[0]
    if task.poll() != 0:
        print(confirm)
        raise ValueError('fail')
    task = shell(
        f"wget -c --load-cookies /tmp/cookies.txt 'https://docs.google.com/uc?export=download&confirm={confirm}&id={fid}' -O {fname}",
        block=False)
    return task


def to_json_format(obj):
    if isinstance(obj, np.ndarray):
        # return obj.tolist()
        return obj
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [to_json_format(subobj) for subobj in obj]
    if isinstance(obj, dict):
        for key in obj.keys():
            obj[key] = to_json_format(obj[key])
        return obj
    if isinstance(obj, int) or isinstance(obj, str) or isinstance(obj, float):
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    return obj
