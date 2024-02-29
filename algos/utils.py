import torch
import random
import numpy as np


def to(*args, device='cpu', dtype=torch.float):
    """ loads tuple of tensors onto device and casts to dtype """
    vals = []
    for arg in args:
        if len(arg.shape) == 1:
            arg = arg.unsqueeze(1)
        if arg.dtype is torch.bool:
            vals += [arg.to(device)]
        else:
            vals += [arg.type(dtype).to(device)]
    return tuple(vals)


def seed_all(seed=None):
    """
    Seed the random number generators in Python, NumPy, and PyTorch.

    Parameters:
        seed (int or None): The seed to use. If None, a random seed will be chosen.

    Returns:
        None
    """
    # Seed Python random number generator
    random.seed(seed)

    # Seed NumPy random number generator
    np.random.seed(seed)

    # Seed PyTorch random number generator for CPU
    torch.manual_seed(seed)

    # Seed PyTorch random number generator for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def polyak_update(critic, ema_critic, critic_ema_decay=0.98):

    """

    :param critic: critic to source the weights from (ie the critic we are training with grad descent)
    :param ema_critic: critic that will be used to compute target values
    :param critic_ema_decay: smoothing factor, default 0.98
    :return: None
    """
    with torch.no_grad():
        for critic_params, ema_critic_params in zip(critic.parameters(), ema_critic.parameters()):
            ema_critic_params.data = ema_critic_params * critic_ema_decay + (1 - critic_ema_decay) * critic_params