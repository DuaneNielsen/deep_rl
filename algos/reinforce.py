import torch
from buffer import ReplayBufferDataset
from torch.utils.data import DataLoader


def train(buffer, policy_net, optim, clip_min=-2.0, clip_max=-0.1, device='cpu', dtype=torch.float):
    """
    Reinforce

    Args:
        buffer: ReplayBuffer, BUFFER WILL BE CLEARED AFTER TRAINING
        policy_net: function policy(state) -> probability distribution over actions
        optim: optimizer
        clip_min: probs will be clipped to exp(clip_min)
        clip_max: probs will be clipped to exp(clip_max)
        device: to train on
        dtype: to convert to
    """

    ds = ReplayBufferDataset(buffer, fields=('s', 'a'), info_keys=['g'])
    dl = DataLoader(ds, batch_size=10000, num_workers=0)

    for trs in dl:
        state= trs.s.type(dtype).to(device)
        action = trs.a.type(dtype).to(device)
        G = trs.g.type(dtype).to(device).unsqueeze(1)

        optim.zero_grad()
        a_dist = policy_net(state)
        G = (G - torch.mean(G)) / (G.max() - G.min()).detach()
        logprob = a_dist.log_prob(action)
        loss = - torch.mean(logprob.clamp(max=clip_max) * G)
        loss.backward()
        optim.step()

        # break

    """ since this is an on-policy algorithm, throw away the data """
    buffer.clear()
