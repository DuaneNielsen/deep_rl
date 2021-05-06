import yaml
import argparse
from pathlib import Path
import torch
import collections.abc
import re

"""
Config module provides flexible management of configuration

"""


class NullScheduler:
    """ Empty scheduler for use as a placeholder to keep code compatible"""

    def __init__(self):
        pass

    def step(self, *args, **kwargs):
        pass


def _get_kwargs(args, key):
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


def _flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _set_if_not_set(args, dict):
    """
    Sets an argument if it's not already set in the args
    Args:
        args: args namespace
        dict: a dict containing arguments to check
    :return:
    """
    for key, value in dict.items():
        if key in vars(args) and vars(args)[key] is None:
            vars(args)[key] = dict[key]
        elif key not in vars(args):
            vars(args)[key] = dict[key]
    return args


def _counter():
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
    Args:
        args: the configuration Namespace
        parameters: model.parameters()
    Returns:
         optimizer, scheduler

    if scheduler not specified a placeholder scheduler will be returned
    """
    optim_class, optim_kwargs = _get_kwargs(args, 'optim')
    optim_class = getattr(torch.optim, optim_class)
    optim = optim_class(parameters, **optim_kwargs)
    scheduler_class, scheduler_kwargs = _get_kwargs(args, 'scheduler')
    if scheduler_class is None:
        return optim, NullScheduler()
    scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_class)
    scheduler = scheduler_class(optim, **scheduler_kwargs)
    return optim, scheduler


def exists_and_not_none(config, attr):
    """ returns true if the config item exists and is set """
    if hasattr(config, attr):
        if vars(config)[attr] is not None:
            return True
    return False


class EvalAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(EvalAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, eval(values))


class ArgumentParser:
    """ a wrapper around argparse.ArgumentParser enhanced to worth with yaml files"""
    def __init__(self, description=None):
        self.parser = argparse.ArgumentParser(description=description)

    def add_argument(self, *args, **kwargs):
        """ just use like argparse.ArgumentParser.add_argument """
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self, args=None):
        """
        Loads the configuration

        Args:
            parser: an argparse parser with configured command line switches
            args: an optional list of arguments to parse, otherwise will read the command line

        Returns:
            an argparse Namespace with configured arguments

        Args can be taken from 3 places, in precendence

            1.  The value passed by command line switch
            2.  The config file
            3.  The default argument set by add_argument
            4.  else the value will be set to None

        the config file is a yaml file, with nested names being separated by hyphens

        eg:

        .. code-block:: yaml

            comment: hello world
            seed: 0
            env:
              name: CartPoleContinuous-v1
            episodes_per_batch: 8

        will become

        .. code-block:: python

            config.comment = 'hello world'
            config.seed = 0
            config.env-name = 'CartPoleContinuous-v1'
            config.episodes_per_batch = 8

        """

        first_config = self.parser.parse_args(args)

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
        if exists_and_not_none(first_config, 'config'):
            with Path(first_config.config).open() as f:
                conf = yaml.load(f, Loader=loader)
                conf = _flatten(conf)
                self.parser.set_defaults(**conf)

        final_config = self.parser.parse_args(args)

        ''' if run_id not explicitly set, then guess it'''
        if exists_and_not_none(final_config, 'run_id'):
            if final_config.run_id == -1:
                final_config.run_id = _counter()

            vars(final_config)['run_dir'] = f'runs/run_{final_config.run_id}'

        return final_config


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
