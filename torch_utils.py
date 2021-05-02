import torch
from pathlib import Path
from torchvision.io import write_video
import numpy as np
from torch.utils.data import Sampler
from typing import Optional, Sized
from logs import logger


def save_checkpoint(directory, prefix=None, **kwargs):
    """
    save torch.nn Modules to a directory, uses the state_dict() method

    Args:
        directory: the directory to save to, will be created if it doesnt exist
        prefix: prefix to apply to the files to be saved
        kwargs: argument name is taken as the save file name, argument value is a torch.nn.Module to be saved

    .. code-block:: python

        checkpoint.save('runs/run_42', 'best', policy=policy_net, optim=optim)

    will write out files

    .. code-block::

        runs/run_42/best_policy.sd
        runs/run_42/best_optim.sd

    """
    prefix = prefix + '_' if prefix is not None else ''
    Path(directory).mkdir(parents=True, exist_ok=True)
    for key, net in kwargs.items():
        torch.save(net.state_dict(), directory + '/' + prefix + key + '.sd')


def load_checkpoint(directory, prefix=None, **kwargs):
    """
    loads saved weights into a torch.nn.Module using the state_dict() method

    Args:
        directory: the directory to load from
        prefix: a prefix associated with the files
        kwargs: argument name is take to be the file to look for, argument value is a torch.nn.Module to load

    given files

    .. code-block::

        runs/run_42/best_policy.sd
        runs/run_42/best_optim.sd

    the code

    .. code-block:: python

        checkpoint.load('runs/run_42', 'best', policy=policy_net, optim=optim)

    will load the state_dicts from disk for policy_net and optim

    """
    sd = {}
    prefix = prefix + '_' if prefix is not None else ''
    for file in Path(directory).glob(f'{prefix}*.sd'):
        key = file.name[len(prefix):-3]
        sd[key] = file

    for key, net in kwargs.items():
        assert key in sd, f"could not find a file for {key}"
        net.load_state_dict(torch.load(sd[key]))


def write_mp4(file, vid_buffer):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    stream = torch.from_numpy(np.stack(vid_buffer))
    write_video(str(file), stream, 24.0)


class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            yield from torch.randperm(n, generator=self.generator).tolist()

    def __len__(self):
        return self.num_samples


def log_torch_video(logger):
    for key, value in logger.log.items():
        if 'video' in key:
            vid_filename = f'{logger.run_dir}/{key}.mp4'
            write_mp4(vid_filename, logger.log[key])


logger.writers.append(log_torch_video)
