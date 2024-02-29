import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
import buffer as bf
import copy
from algos.utils import polyak_update


class PPOWrapModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.old = copy.deepcopy(model)
        self.new = model

    def forward(self, input, old=False):
        if old:
            return self.old(input)
        else:
            return self.new(input)

    def parameters(self, recurse=True):
        return self.new.parameters(recurse)

    def backup(self):
        self.old.load_state_dict(self.new.state_dict())


def ppo_loss(newlogprob, oldlogprob, advantage, clip=0.2):
    log_ratio = (newlogprob - oldlogprob)
    # clamp the log to stop infinities (85 is for 32 bit floats)
    log_ratio.clamp_(min=-10.0, max=10.0)
    ratio = torch.exp(log_ratio)

    clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip)
    clipped_step = clipped_ratio * advantage
    full_step = ratio * advantage
    min_step = torch.stack((full_step, clipped_step), dim=1)
    min_step, clipped = torch.min(min_step, dim=1)
    return - min_step.mean()


def train(buffer, policy_net, optim, clip=0.2, dtype=torch.float, device='cpu'):
    ds = bf.ReplayBufferDataset(buffer, fields=('s', 'a'), info_keys='g')
    dl = DataLoader(ds, batch_size=10000)

    for state, action, advantage in dl:
        state = state.type(dtype).to(device)
        action = action.type(dtype).to(device)

        # this is just standard normalized policy gradient, not advantage
        advantage = advantage.type(dtype).to(device).unsqueeze(1)
        advantage = (advantage - advantage.mean()) / advantage.var()

        optim.zero_grad()
        new_logprob = policy_net(state).log_prob(action)
        old_logprob = policy_net(state, old=True).log_prob(action).detach()
        loss = ppo_loss(new_logprob, old_logprob, advantage, clip=clip)
        policy_net.backup()
        loss.backward()
        optim.step()

        buffer.clear()
        break

def td_targets(bootstrap_value, rewards, done, discount):
    v_targets = torch.zeros_like(rewards)
    prev = bootstrap_value

    for i in reversed(range(0, len(rewards))):
        prev = rewards[i] + discount * prev * (~done[i]).float()
        v_targets[i] = prev

    return v_targets


def train_a2c(dl, a2c_net, optim, discount=0.95, clip=0.2, batch_size=64, device='cpu', precision=torch.float):
    """

    Advantage Actor Critic

    Args:
        buffer: replay buffer
        a2c_net: a2c_net(state) -> values, a_dist
        optim: optimizer for a2c_net
        discount: discount factor, default 0.95
        batch_size: batch size
        device: device to train on
        precision: all floats will be cast to dtype

    """

    """ loads 1 batch and runs a single training step """
    for s, a, s_p, r, d in dl:
        state = s.type(precision).to(device)
        action = a.type(precision).to(device).squeeze()
        state_p = s_p.type(precision).to(device)
        r = r.type(precision).to(device).unsqueeze(1)
        d = d.to(device).unsqueeze(1)

        optim.zero_grad()

        v_s, a_dist = a2c_net(state)

        with torch.no_grad():
            v_sp, _ = a2c_net(state_p)
            td_tar = td_targets(v_sp[-1, :], r, d, discount)
            advantage = r + v_sp * discount * (~d).float() - v_s

        critic_loss = mse_loss(td_tar, v_s)

        new_logprob = a_dist.log_prob(action)
        _, old_dist = a2c_net(state, old=True)
        old_logprob = old_dist.log_prob(action).detach()
        actor_loss = ppo_loss(new_logprob, old_logprob, advantage.squeeze(), clip=clip)
        a2c_net.backup()

        loss = actor_loss + 0.5 * critic_loss - 0.01 * a_dist.entropy().mean()

        loss.backward()
        optim.step()

        break


def train_a2c_stable(dl, value_net, value_optim, policy_net, policy_optim, discount=0.95, clip=0.2, batch_size=64,
                     device='cpu', precision=torch.float):
    """

    Advantage Actor Critic

    Args:
        buffer: replay buffer
        a2c_net: a2c_net(state) -> values, a_dist
        optim: optimizer for a2c_net
        discount: discount factor, default 0.95
        batch_size: batch size
        device: device to train on
        precision: all floats will be cast to dtype

    """

    """ loads 1 batch and runs a single training step """
    for s, a, s_p, r, d in dl:
        state = s.type(precision).to(device)
        action = a.type(precision).to(device).squeeze()
        state_p = s_p.type(precision).to(device)
        r = r.type(precision).to(device).unsqueeze(1)
        d = d.to(device).unsqueeze(1)

        value_optim.zero_grad()
        policy_optim.zero_grad()

        v_s, a_dist = value_net(state), policy_net(state)

        with torch.no_grad():
            v_sp = value_net(state_p)
            td_tar = td_targets(v_sp[-1, :], r, d, discount)
            advantage = r + v_sp * discount * (~d).float() - v_s

        critic_loss = mse_loss(td_tar, v_s)
        critic_loss.backward()
        value_optim.step()

        new_logprob = a_dist.log_prob(action)
        old_dist = policy_net(state, old=True)
        old_logprob = old_dist.log_prob(action).detach()
        actor_loss = ppo_loss(new_logprob, old_logprob, advantage.squeeze(), clip=clip)
        policy_net.backup()

        loss = actor_loss - 0.01 * a_dist.entropy().mean()

        loss.backward()
        policy_optim.step()

        break