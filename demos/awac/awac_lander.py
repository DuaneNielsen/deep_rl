import torch
import torch.nn as nn

import gym

import driver

from torch.distributions import Categorical
from argparse import ArgumentParser
import checkpoint
from torch.nn.functional import log_softmax
import random


if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='configuration switches')

    """ reproducibility """
    parser.add_argument('--seed', type=int, default=None)

    config = parser.parse_args()

    config.env_name = 'LunarLander-v2'
    config.hidden_dim = 64
    config.device = 'cpu'

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)
        random.seed(config.seed)

    """ environment """
    def make_env():
        env = gym.make(config.env_name)
        if config.seed is not None:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
        return env

    """ training env with replay buffer """
    env = make_env()

    """ network """
    class AWACnet(nn.Module):
        """

        """
        def __init__(self, input_dims, actions, hidden_dims):
            super().__init__()
            self.q = nn.Sequential(nn.Linear(input_dims, hidden_dims), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dims, actions))
            self.policy = nn.Sequential(nn.Linear(input_dims, hidden_dims), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dims, actions))

        def forward(self, state):
            values = self.q(state)
            actions = self.policy(state)
            action_dist = Categorical(logits=log_softmax(actions, dim=1))
            return values, action_dist

    awac_net = AWACnet(
        input_dims=env.observation_space.shape[0],
        actions=env.action_space.n,
        hidden_dims=config.hidden_dim).to(config.device)

    """ load weights from file if required"""
    checkpoint.load('.', prefix='lander2', awac_net=awac_net)

    """ policy to run on environment """
    def policy(state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(config.device)
        value, action = awac_net(state)
        a = action.sample()
        assert torch.isnan(a) == False
        return a.item()

    """ demo  """
    while True:
        driver.episode(env, policy, render=True)