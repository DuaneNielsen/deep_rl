import torch
from torch import nn
from torch.utils.data import DataLoader
import buffer as bf
import copy


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