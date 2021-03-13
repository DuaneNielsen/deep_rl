import torch
import torch.nn as nn

import gym
import env

import driver
from gymviz import Plot

import buffer as bf
from algos import ppo
from distributions import ScaledTanhTransformedGaussian
from config import exists_and_not_none, ArgumentParser
import wandb
import wandb_utils
import checkpoint
import baselines.helper as helper

if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='configuration switches')

    """ reproducibility """
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plot', action='store_true', default=False)

    config = parser.parse_args()

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)

    """ environment """
    def make_env():
        env = gym.make('Pendulum-v0')
        if config.seed is not None:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
        return env

    """ test env """
    test_env = make_env()
    if config.plot:
        test_env = Plot(test_env, episodes_per_point=1, title=f'ppo-a2c-Pendulum-v0')
    evaluator = helper.Evaluator(test_env)

    """ network """
    class A2CNet(nn.Module):
        """
        policy(state) returns distribution over actions
        uses ScaledTanhTransformedGaussian as per Hafner
        """
        def __init__(self, input_dims, hidden_dims, min, max):
            super().__init__()
            self.min = min
            self.max = max
            self.mu = nn.Sequential(nn.Linear(input_dims, hidden_dims), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dims, 2))
            self.scale = nn.Linear(input_dims, 1, bias=False)

        def forward(self, state):
            output = self.mu(state)
            value = output[..., 0:1]
            mu = output[..., 1:2]
            scale = torch.sigmoid(self.scale(state)) + 0.01
            a_dist = ScaledTanhTransformedGaussian(mu, scale, min=self.min, max=self.max)
            return value, a_dist

    a2c_net = A2CNet(
        input_dims=test_env.observation_space.shape[0],
        hidden_dims=16,
        min=test_env.action_space.low[0],
        max=test_env.action_space.high[0])

    a2c_net = ppo.PPOWrapModel(a2c_net)

    checkpoint.load('.', prefix='pendulum', a2c_net=a2c_net)

    """ policy to run on environment """
    def policy(state):
        state = torch.from_numpy(state).float()
        value, action = a2c_net(state)
        a = action.rsample()
        assert torch.isnan(a) == False
        return a.numpy()

    """ demo  """
    evaluator.demo(True, policy)