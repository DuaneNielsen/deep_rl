import torch
import buffer as bf
from torch.utils.data import DataLoader, SubsetRandomSampler
import random


def train(buffer, v_net, optim, batch_size, discount, device='cpu', dtype=torch.float):
    """
    Trains a value function using temporal difference

    Args:
        buffer: replay buffer
        v_net: v_net(state) -> value
        optim: optimizer for v_net
        batch_size: batch_size
        discount: discount
        device: device to load data to
        dtype: cast floating point data to dtype

    """

    """ sample from batch_size transitions from the replay buffer """
    ds = bf.ReplayBufferDataset(buffer)
    sampler = SubsetRandomSampler(random.sample(range(len(ds)), batch_size))
    dl = DataLoader(buffer, batch_size=batch_size, sampler=sampler)

    """ loads 1 batch and runs a single training step """
    for s, a, s_p, r, d in dl:
        s, s_p = s.type(dtype).to(device), s_p.type(dtype).to(device)
        r, d = r.type(dtype).to(device).unsqueeze(1), d.to(device).unsqueeze(1)

        optim.zero_grad()

        v_s = v_net(s)
        with torch.no_grad():
            v_sp = v_net(s_p) * (~d).float()
            v_sp = r + v_sp * discount
        loss = torch.mean(v_sp - v_s) ** 2

        loss.backward()
        optim.step()
