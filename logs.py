import numpy as np
from statistics import mean, stdev


class Logger:
    def __init__(self):
        self.log = {}
        self.writers = []
        self.run_dir = None

    def write(self):
        for writer in self.writers:
            writer(self)
        self.log = {}


logger = Logger()


def init(run_dir):
    logger.run_dir = run_dir


def log(log_dict):
    logger.log.update(log_dict)


def write():
    logger.write()


def tensor_stats(name, tensor):
    st = {}
    st[name + ' Mean'] = tensor.mean().item()
    st[name + ' Std'] = tensor.std().item()
    st[name + ' Max'] = tensor.max().item()
    st[name + ' Min'] = tensor.min().item()
    st[name + ' histogram'] = tensor.detach().cpu().numpy()
    return st


def numpy_stats(name, ndarray):
    st = {}
    st[name + ' Mean'] = np.mean(ndarray)
    st[name + ' Std'] = np.std(ndarray)
    st[name + ' Max'] = np.max(ndarray)
    st[name + ' Min'] = np.min(ndarray)
    st[name + ' histogram'] = ndarray
    return st


def list_stats(name, lst):
    st = {}
    st[name + ' Mean'] = mean(lst)
    st[name + ' Std'] = stdev(lst)
    st[name + ' Max'] = max(lst)
    st[name + ' Min'] = min(lst)
    st[name + ' histogram'] = lst
    return st
