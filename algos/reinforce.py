import torch
from torch.utils.data import DataLoader
from collections import namedtuple


def to(*args, device='cpu', dtype=torch.float):
    return tuple([arg.type(dtype).to(device) for arg in args])


FastTuple = namedtuple('FastTuple', ['s', 'a', 'i'])


def fast_collate(data):
    state = []
    action = []
    G = []
    for s, i, a, s_p, r, d, i_p in data:
        state += [s]
        action += [a]
        G += [[i['g']]]
    s = torch.tensor(state)
    a = torch.tensor(action)
    g = torch.tensor(G)

    return FastTuple(s=s, a=a, i={'g': g})


def train(buffer, policy_net, optim, device='cpu', dtype=torch.float):

    dl = DataLoader(buffer, batch_size=10000, num_workers=8, collate_fn=fast_collate)

    for transitions in dl:
        state, action, G = to(transitions.s, transitions.a, transitions.i['g'], device=device, dtype=dtype)
        optim.zero_grad()
        a_dist = policy_net(state)
        G = (G - torch.mean(G)) / (G.max() - G.min()).detach()
        loss = - torch.mean(a_dist.log_prob(action).clamp(min=-2.0, max=-0.1) * G)
        loss.backward()
        optim.step()