import torch


def to(*args, device='cpu', dtype=torch.float):
    vals = []
    for arg in args:
        if len(arg.shape) == 1:
            arg = arg.unsqueeze(1)
        if arg.dtype is torch.bool:
            vals += [arg.to(device)]
        else:
            vals += [arg.type(dtype).to(device)]
    return tuple(vals)
