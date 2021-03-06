import torch
import torch.nn as nn

import gym

import buffer as bf
import driver
from algos import reinforce
from distributions import ScaledTanhTransformedGaussian
import env
from gymviz import Plot
import wandb
import wandb_utils
from config import load_config
import argparse


class RescaleReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return reward / 200.0


if __name__ == '__main__':

    """ configuration """
    parser = argparse.ArgumentParser(description='configuration switches')

    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('-d', '--device', type=str)
    parser.add_argument('-r', '--run_id', type=int, default=-1)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--epochs_per_test', type=int, default=500)
    parser.add_argument('--seed', type=int, default=None)

    """ resume settings """
    parser.add_argument('--demo', )
    parser.add_argument('-l', '--load', type=str, default=None)

    """ environment """
    parser.add_argument('--env_name', type=str, default='CartPoleContinuous-v1')

    """ visualization params """
    parser.add_argument('--render', action='store_true', default=False)

    """ hyper-parameters """
    parser.add_argument('--optim-class', type=str)
    parser.add_argument('--optim-lr', type=float)
    parser.add_argument('--scheduler_class', type=str)
    parser.add_argument('--episodes_per_batch', type=int, default=16)
    parser.add_argument('--discount', type=int, default=0.95)
    parser.add_argument('--min_variance', type=float, default=0.05)
    parser.add_argument('--hidden_dim', type=int, default=16)

    config = load_config(parser)

    wandb.init(project="reinforce-cartpole", config=config)

    """ environment """
    env = gym.make(config.env_name)
    env = RescaleReward(env)  # rescale reward to reduce gradient size
    assert len(env.observation_space.shape) == 1, "Only 1D continuous observation space is supported"
    assert len(env.action_space.shape) == 1, "Only 1D continuous action space is supported"

    """ replay buffer """
    env, buffer = bf.wrap(env)
    buffer.attach_enrichment(bf.DiscountedReturns(discount=config.discount))
    env = Plot(env, episodes_per_point=config.episodes_per_batch)
    env = wandb_utils.LogRewards(env)

    """ random seed """
    if config.seed:
        torch.manual_seed(config.seed)
        env.seed(config.seed)
        env.action_space.seed(config.seed)

    """ network """
    class PolicyNet(nn.Module):
        """
        policy(state) returns distribution over actions
        uses ScaledTanhTransformedGaussian as per Hafner
        """

        def __init__(self, input_dim, hidden_dim, output_dim=1, min=-1.0, max=1.0):
            super().__init__()
            self.mu = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ELU(inplace=True),
                                    nn.Linear(hidden_dim, output_dim))
            self.scale = nn.Linear(input_dim, output_dim, bias=False)
            self.min = min
            self.max = max

        def forward(self, state):
            mu = self.mu(state)
            scale = torch.sigmoid(self.scale(state)) + config.min_variance
            return ScaledTanhTransformedGaussian(mu, scale, min=self.min, max=self.max)


    policy_net = PolicyNet(env.observation_space.shape[0], config.hidden_dim, env.action_space.shape[0])
    optim = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
    wandb.watch(policy_net)

    """ policy to run on environment """
    def policy(state):
        state = torch.from_numpy(state).float()
        action = policy_net(state)
        a = action.rsample().numpy()
        return a

    """ sample """
    for epoch in range(8000):
        for ep in range(config.episodes_per_batch):
            driver.episode(env, policy, render=config.render)

        reinforce.train(buffer, policy_net, optim)

        """ test loop goes here """
        if epoch % 1:
            pass