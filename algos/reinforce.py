import torch
from buffer import ReplayBufferDataset
from torch.utils.data import DataLoader
import algos.utils


def train(buffer, policy_net, optim, device='cpu', dtype=torch.float):
    """
    REINFORCE

    param: buffer - ReplayBuffer, BUFFER WILL BE CLEARED AFTER TRAINING
    param: policy_net - function policy(state) returns probability distribution over actions
    param: optim - optimizer
    device: device to train on
    dtype: dtype to convert to
    """

    ds = ReplayBufferDataset(buffer, fields=('s', 'a'), info_keys=['g'])
    dl = DataLoader(ds, batch_size=10000, num_workers=0)

    for transitions in dl:
        state, action, G = algos.utils.to(transitions.s, transitions.a, transitions.g, device=device, dtype=dtype)
        optim.zero_grad()
        a_dist = policy_net(state)
        G = (G - torch.mean(G)) / (G.max() - G.min()).detach()
        loss = - torch.mean(a_dist.log_prob(action).clamp(min=-2.0, max=-0.1) * G)
        loss.backward()
        optim.step()

    """ since this is an on-policy algorithm, throw away the data """
    buffer.clear()
