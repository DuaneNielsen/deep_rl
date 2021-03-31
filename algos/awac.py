from random import sample, randint

import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, SubsetRandomSampler
import buffer as bf
import random
import time
from statistics import mean, stdev
import wandb
from torch.distributions.kl import kl_divergence

timing = []


def train_discrete(dl, a2c_net, critic_optim, actor_optim, discount=0.95, lam=0.3, device='cpu',
               debug=False, measure_kl=False, global_step=None):
    """

    AWAC


    """

    begin = time.time()
    loaded = None
    end = None
    init = None

    if debug:
        init = time.time()

    for s, a, s_p, r, d in dl:
        state = s.to(device)
        action = a.to(device)
        state_p = s_p.to(device)
        reward = r.to(device)
        done = d.to(device)

        if debug:
            loaded = time.time()

        critic_optim.zero_grad()
        actor_optim.zero_grad()

        N = state.shape[0]

        q_s, a_dist = a2c_net(state)
        v_sa = q_s[torch.arange(N), action.squeeze()].unsqueeze(1)

        with torch.no_grad():
            q_sp, a_sp_dist = a2c_net(state_p)
            v_s = torch.sum(q_s.detach() * a_dist.probs.detach(), dim=1, keepdim=True)
            v_sp = torch.sum(q_sp * a_sp_dist.probs, dim=1, keepdim=True)
            target = reward + v_sp * discount * done
            advantage = v_sa - v_s

        critic_loss = mse_loss(target, v_sa)

        action_logprob = a_dist.log_prob(action.squeeze()).unsqueeze(1)
        actor_loss = - torch.mean(action_logprob * torch.exp(advantage / lam))

        #entropy = torch.mean(- action_logprob * torch.exp(action_logprob))

        critic_loss.backward()
        critic_optim.step()
        actor_loss.backward()
        actor_optim.step()

        if measure_kl:
            # compute kl divergence after update
            with torch.no_grad():
                new_a_dist = a2c_net.policy(state)
                div = kl_divergence(a_dist, new_a_dist)
                kl_mean = div.mean().item()
                kl_std = div.std().item()
                wandb.log({'kl_mean': kl_mean, 'kl_std': kl_std, 'global_step': global_step})

        if debug:
            # if torch.stack([torch.isnan(p.grad).any() for p in a2c_net.parameters()]).any():
            #     print(a2c_net.parameters())
            #     assert False, "NaN detected"

            end = time.time()
            init_time = init - begin
            load_time = loaded - init
            train_time = end - loaded
            timing.append((init_time, load_time, train_time))

            if len(timing) % 100 == 0 and len(timing) > 1:
                mean_init = mean([init_time for init_time, load_time, train_time in timing])
                mean_load = mean([load_time for _, load_time, train_time in timing])
                mean_train = mean([train_time for _, load_time, train_time in timing])
                print(mean_init, mean_load, mean_train)

        break


class FastOfflineDataset:
    def __init__(self, load_buff, capacity, device='cpu', length=None, rescale_reward=1.0):

        if length is None:
            length = len(load_buff)

        self.device = device

        s, a, s_p, r, d = load_buff[0]

        self.length = min(len(load_buff), length)
        self.capacity = max(length, capacity)
        self.state = torch.empty(self.capacity, *s.shape, dtype=torch.float32, device=device)
        self.action = torch.empty(self.capacity, 1, dtype=torch.long, device=device)
        self.state_p = torch.empty(self.capacity, *s_p.shape, dtype=torch.float32, device=device)
        self.reward = torch.empty(self.capacity, 1, dtype=torch.float32, device=device)
        self.done = torch.empty(self.capacity, 1, dtype=torch.float32, device=device)

        for i, sampled in enumerate(sample(range(len(load_buff)), self.length)):
            self[i] = load_buff[sampled]

        self.reward = self.reward * rescale_reward

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.state[i], self.action[i], self.state_p[i], self.reward[i], self.done[i]

    def __setitem__(self, i, transition):
        s, a, s_p, r, d = transition
        self.state[i] = torch.from_numpy(s).type(torch.float32)
        self.action[i] = a
        self.state_p[i] = torch.from_numpy(s_p).type(torch.float32)
        self.reward[i] = r
        self.done[i] = 0.0 if d else 1.0

    def append(self, transition):
        # if capacity in the buffer, append, else overwrite at random
        if self.length < self.capacity:
            i = self.length
            self.length += 1
        else:
            i = randint(0, self.length-1)
        self[i] = transition