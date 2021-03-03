import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import buffer as bf


def train(buffer, v_net, v_optim, policy_net, policy_optim, discount=0.95, batch_size=10000, device='cpu', dtype=torch.float):

    """ sample from batch_size transitions from the replay buffer """
    ds = bf.ReplayBufferDataset(buffer)
    sampler = SubsetRandomSampler(random.sample(range(len(ds)), min(batch_size, len(ds))))
    dl = DataLoader(buffer, batch_size=batch_size, sampler=sampler)

    """ loads 1 batch and runs a single training step """
    for s, a, s_p, r, d in dl:
        state = s.type(dtype).to(device)
        action = a.type(dtype).to(device)
        state_p = s_p.type(dtype).to(device)
        r = r.type(dtype).to(device).unsqueeze(1)
        d = d.to(device).unsqueeze(1)

        v_optim.zero_grad()

        v_s = v_net(state)
        with torch.no_grad():
            v_sp = v_net(state_p) * (~d).float()
            v_sp = r + v_sp * discount
        loss = torch.mean((v_sp - v_s) ** 2)

        loss.backward()
        v_optim.step()

        with torch.no_grad():
            advantage = r + v_net(state_p) - v_net(state)

        policy_optim.zero_grad()

        a_dist = policy_net(state)
        loss = - torch.mean(a_dist.log_prob(action).clamp(min=-2.0, max=-0.1) * advantage)

        loss.backward()
        policy_optim.step()

        break

    """ since this is an on-policy algorithm, throw away the data """
    buffer.clear()

