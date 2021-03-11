import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import buffer as bf


def train(buffer, a2c_net, optim, discount=0.95, batch_size=10000, device='cpu', dtype=torch.float):
    """

    Advantage Actor Critic

    Args:
        buffer: replay buffer
        a2c_net: a2c_net(state) -> values, a_dist
        optim: optimizer for a2c_net
        policy_net: policy_net(state) -> action
        policy_optim: optimizer for policy
        discount: discount factor, default 0.95
        batch_size: batch size
        device: device to train on
        dtype: all floats will be cast to dtype

    """

    """ sample from batch_size transitions from the replay buffer """
    ds = bf.ReplayBufferDataset(buffer)
    dl = DataLoader(buffer, batch_size=batch_size)

    """ loads 1 batch and runs a single training step """
    for s, a, s_p, r, d in dl:
        state = s.type(dtype).to(device)
        action = a.type(dtype).to(device)
        state_p = s_p.type(dtype).to(device)
        r = r.type(dtype).to(device).unsqueeze(1)
        d = d.to(device).unsqueeze(1)


        optim.zero_grad()

        v_s, a_dist = a2c_net(state)
        #with torch.no_grad():
        tail_value, _ = a2c_net(state_p)
        G = torch.zeros_like(r)
        G[-1, :] = tail_value[-1:]

        for i in reversed(range(0, len(r) - 1)):
            G[i] += r[i] + discount * G[i + 1] * (~d[i + 1]).float()
        v_sp = G
        #v_sp = v_sp * (~d).float()
        advantage = r + v_sp * discount - v_s
        critic_loss = mse_loss(r + v_sp * discount, v_s)

        action_logprob = a_dist.log_prob(action)
        actor_loss = - torch.mean(action_logprob.clamp(max=-0.3) * advantage)

        entropy = torch.mean(- action_logprob * torch.exp(action_logprob))

        loss = actor_loss + 0.5 * critic_loss - 0.05 * entropy

        loss.backward()
        optim.step()

        break

    """ since this is an on-policy algorithm, throw away the data """
    buffer.clear()

