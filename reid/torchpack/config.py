import sys
from importlib import import_module
from os import path


def load_cfg(cfg_file):
    sys.path.append(path.dirname(cfg_file))
    module_name = path.basename(cfg_file).rstrip('.py')
    cfg = import_module(module_name)
    return cfg
