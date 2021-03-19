import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, SubsetRandomSampler
import buffer as bf
import random


def train_discrete(buffer, a2c_net, critic_optim, actor_optim, discount=0.95, batch_size=64, device='cpu', dtype=torch.float):
    """

    AWAC


    """

    """ sample from batch_size transitions from the replay buffer """
    ds = bf.ReplayBufferDataset(buffer)
    sampler = SubsetRandomSampler(random.sample(range(len(ds)), batch_size))
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler)

    """ loads 1 batch and runs a single training step """
    for s, a, s_p, r, d in dl:
        state = s.type(dtype).to(device)
        action = a.to(device)
        state_p = s_p.type(dtype).to(device)
        r = r.type(dtype).to(device).unsqueeze(1)
        done = (~d.to(device)).float().unsqueeze(1)

        critic_optim.zero_grad()
        actor_optim.zero_grad()

        N = state.shape[0]

        q_s, a_dist = a2c_net(state)
        v_s = q_s[torch.arange(N), action.squeeze()].unsqueeze(1)

        with torch.no_grad():
            q_sp, a_sp_dist = a2c_net(state_p)
            v_sp = torch.sum(q_sp * a_sp_dist.probs, dim=1, keepdim=True)
            target = r + v_sp * discount * done
            advantage = target - v_s

        critic_loss = mse_loss(target, v_s)

        action_logprob = a_dist.log_prob(action.squeeze()).unsqueeze(1)
        actor_loss = - torch.mean(action_logprob * advantage)

        #entropy = torch.mean(- action_logprob * torch.exp(action_logprob))

        critic_loss.backward()
        critic_optim.step()
        actor_loss.backward()
        actor_optim.step()

        break