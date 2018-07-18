from __future__ import print_function, absolute_import

from torch.nn import Parameter

import lz, json
from lz import *


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_p(osp.dirname(fpath), delete=False)
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, is_best, fpath='checkpoint.pth'):
    mkdir_p(osp.dirname(fpath), delete=False)
    torch.save(state, fpath)
    dest = osp.join(osp.dirname(fpath), 'model_best.pth')
    if is_best or not osp.exists(dest):
        shutil.copy(fpath, dest)


def load_checkpoint(fpath, map_location=None):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath, map_location=map_location)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


def load_state_dict(model, state_dict, own_prefix='', own_de_prefix=''):
    own_state = model.state_dict()
    success = []
    for name, param in state_dict.items():
        if 'base_model.' + name in own_state:
            name = 'base_model.' + name
        if 'module.' + name in own_state:
            name = 'module.' + name
        if 'base_model.base_model.' + name in own_state:
            name = 'base_model.base_model.' + name
        if own_prefix + name in own_state:
            name = own_prefix + name
        if name.replace(own_de_prefix, '') in own_state:
            name = name.replace(own_de_prefix, '')

        if name not in own_state:
            print('ignore key "{}" in his state_dict'.format(name))
            continue

        if isinstance(param, nn.Parameter):
            param = param.data

        if own_state[name].size() == param.size():
            own_state[name].copy_(param)
            # print('{} {} is ok '.format(name, param.size()))
            success.append(name)
        else:
            try:
                if own_state[name].size(0) != param.size(0):
                    param = param.view(8, param.size(0) // 8, param.size(1), param.size(2), param.size(3)).mean(dim=0)
                    # if own_state[name].size(1) != param.size(1):
                    #     param = param.view(param.size(0), 8, param.size(1) // 8, param.size(2), param.size(3)).mean(dim=1)
                    own_state[name].copy_(param)
                else:
                    param = F.pad(param,
                                  (0, own_state[name].size(2) - param.size(2),
                                   0, own_state[name].size(3) - param.size(3)))
                    own_state[name].copy_(param.data)
            except:
                lz.logging.error('fk!!!')
            lz.logging.error('dimension mismatch for param "{}", in the model are {}'
                             ' and in the checkpoint are {}, ...'.format(
                name, own_state[name].size(), param.size()))

    missing = set(own_state.keys()) - set(success)
    if len(missing) > 0:
        print('missing keys in my state_dict: "{}"'.format(missing))


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model
