import torch
import torch.nn as nn
from torch.distributions import Categorical
from logs import logger
from torch.nn.functional import cross_entropy


def train_continuous(dl, policy_net, optim, clip_max=-0.1, clip_min=None,
                     device='cpu', precision=torch.float32,
                     log=False):

    """
    Behavioural Cloning

    Performs 1 epoch of training on the dataset

    Args:
        dl: pytorch dataloader for the dataset
        policy_net: function policy(state) -> probability distribution over actions
        optim: optimizer
        clip_max: probs will be clipped < exp(clip_max)
        clip_min: probs will be clipped > exp(clip_min)
        device: to train on
        precision: to use
    """

    for s, a, s_p, r, d in dl:
        s = s.type(precision).to(device)
        a = a.type(precision).to(device)

        optim.zero_grad()
        a_dist = policy_net(s)
        logprob = a_dist.log_prob(a)
        loss = - torch.mean(logprob.clamp(max=clip_max, min=clip_min))
        loss.backward()
        optim.step()

        if log:
            logger.log['trainer-policy-loss'] = loss.item()
            logger.log['trainer-policy-entropy'] = a_dist.entropy()


def train_discrete(dl, policy_net, optim,
                   device='cpu', precision=torch.float32, log=False):
    """
    Behavioural Cloning

    Performs 1 epoch of training on the dataset

    Args:
        dl: pytorch dataloader for the dataset
        policy_net: function policy(state) -> raw unormalized scores for each action
        optim: optimizer
        device: to train on
        precision: to use
    """

    for s, a, s_p, r, d in dl:
        s = s.type(precision).to(device)
        a = a.to(device)

        optim.zero_grad()
        a_ = policy_net(s)
        loss = cross_entropy(a_, a)
        loss.backward()
        optim.step()

        print(loss.item())

        if log:
            logger.log['trainer-policy-loss'] = loss.item()
            logger.log['trainer-policy-entropy'] = Categorical(a_, ).entropy()
