import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import algos.utils
from buffer import ReplayBufferDataset


def train(buffer, q_net, optim, batch_size, discount, device='cpu', dtype=torch.float):
    """
    Deep Q Network

    buffer: replay buffer
    q_net: nn.Module with forward(state) = values,
                s is (N, S) tensor
                v is (N, A) tensor
        S is the dimensions of the state space, and A is the number of discrete actions
        in english: the q q_net function takes in the state, and returns the value of each action
    optim: optimizer for q_net
    batch: size, the number of transitions to sample each batch step
    discount: the discount for the value function
    device: the device to train on
    dtype: the dtype to convert to
    """

    """ sample from batch_size transitions from the replay buffer """
    ds = ReplayBufferDataset(buffer)
    sampler = SubsetRandomSampler(random.sample(range(len(ds)), batch_size))
    dl = DataLoader(buffer, batch_size=batch_size, sampler=sampler)

    """ loads 1 batch and runs a single training step """
    for s, a, s_p, r, d in dl:
        s, s_p, r, d = algos.utils.to(s, s_p, r, d, device=device, dtype=dtype)
        a = a.to(device)
        N = s.shape[0]

        optim.zero_grad()
        v_s = q_net(s)[torch.arange(N), a].unsqueeze(1)
        with torch.no_grad():
            v_sp, _ = torch.max(q_net(s_p), dim=1, keepdim=True)
            v_sp[d] = 0.0
            v_sp = r + v_sp * discount
        loss = torch.mean((v_sp - v_s) ** 2)

        loss.backward()
        optim.step()
