import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, SubsetRandomSampler
import buffer as bf
import random
import time
from statistics import mean, stdev


def train_discrete(buffer, a2c_net, critic_optim, actor_optim, discount=0.95, batch_size=64, device='cpu', dtype=torch.float):
    """

    AWAC


    """

    """ sample from batch_size transitions from the replay buffer """
    ds = bf.ReplayBufferDataset(buffer)
    sampler = SubsetRandomSampler(random.sample(range(len(ds)), batch_size))
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler, pin_memory=True)

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
        actor_loss = - torch.mean(action_logprob * torch.exp(advantage / 0.3))

        #entropy = torch.mean(- action_logprob * torch.exp(action_logprob))

        critic_loss.backward()
        critic_optim.step()
        actor_loss.backward()
        actor_optim.step()

        break


timing = []


def train_fast(dl, a2c_net, critic_optim, actor_optim, discount=0.95, batch_size=64, lam=0.3, device='cpu', dtype=torch.float, debug=False):
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