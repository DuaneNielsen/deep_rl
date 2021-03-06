import yaml
import argparse
from pathlib import Path
import torch
import collections.abc
import re


"""
Config module provides fleximble managment of configuration

"""


class NullScheduler:
    """ Empty scheduler for use as a placeholder to keep code compatible"""
    def __init__(self):
        pass

    def step(self, *args, **kwargs):
        pass


def get_kwargs(args, key):
    args_dict = vars(args).copy()
    if key + '_class' not in args_dict:
        return None, None
    clazz = args_dict[key + '_class']
    del args_dict[key + '_class']

    kwargs = {}
    for k, v in args_dict.items():
        if k.startswith(key):
            left, right = k.split('_', 1)
            if left == key:
                kwargs[right] = v
    return clazz, kwargs


def flatten(d, parent_key='', sep='-'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def set_if_not_set(args, dict):
    """
    Sets an argument if it's not already set in the args
    :param args: args namespace
    :param dict: a dict containing arguments to check
    :return:
    """
    for key, value in dict.items():
        if key in vars(args) and vars(args)[key] is None:
            vars(args)[key] = dict[key]
        elif key not in vars(args):
            vars(args)[key] = dict[key]
    return args


def counter():
    """
    counter to keep track of run id
    creates a file .run_id in the current directory which stores the most recent id
    """
    run_id_pid = Path('./.run_id')
    count = 1
    if run_id_pid.exists():
        with run_id_pid.open('r+') as f:
            last_id = int(f.readline())
            last_id += 1
            count = last_id
            f.seek(0)
            f.write(str(last_id))
    else:
        with run_id_pid.open('w+') as f:
            f.write(str(count))
    return count


def get_optim(args, parameters):
    """
    Reads the configuration and constructs a scheduler and optimizer
    :param args: the configuration Namespace
    :param parameters: model.parameters()
    :return: optimizer, scheduler
    if scheduler not specified a placeholder scheduler will be returned
    """
    optim_class, optim_kwargs = get_kwargs(args, 'optim')
    optim_class = getattr(torch.optim, optim_class)
    optim = optim_class(parameters, **optim_kwargs)
    scheduler_class, scheduler_kwargs = get_kwargs(args, 'scheduler')
    if scheduler_class is None:
        return optim, NullScheduler()
    scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_class)
    scheduler = scheduler_class(optim, **scheduler_kwargs)
    return optim, scheduler


def load_config(parser, args=None):
    """
    Reads the command switches and creates a config
    Command line switches override config files
    :return: a Namespace of args
    """
    args = parser.parse_args(args)

    """ 
    required due to https://github.com/yaml/pyyaml/issues/173
    pyyaml does not correctly parse scientific notation 
    """
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    """ read the config file """
    if args.config is not None:
        with Path(args.config).open() as f:
            conf = yaml.load(f, Loader=loader)
            conf = flatten(conf)
            args = set_if_not_set(args, conf)

    """ args not set will be set to a default value """
    global_defaults = {
        'optim_class': 'Adam',
        'optim_lr': 1e-4
    }

    args = set_if_not_set(args, global_defaults)

    ''' if run_id not explicitly set, then guess it'''
    if args.run_id == -1:
        args.run_id = counter()

    return args