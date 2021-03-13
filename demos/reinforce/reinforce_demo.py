import torch
import torch.nn as nn
import gym
import env
from distributions import ScaledTanhTransformedGaussian
from argparse import ArgumentParser
import checkpoint
import driver
import gymviz

if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='Cartpole demo')
    parser.add_argument('--plot', action='store_true', default=False)

    """ reproducibility """
    parser.add_argument('--seed', type=int, default=None)

    config = parser.parse_args()

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)

    """ environment """
    env = gym.make('CartPoleContinuous-v1')

    if config.seed is not None:
        env.seed(config.seed)
        env.action_space.seed(config.seed)

    if config.plot:
        env = gymviz.Plot(env)

    """ network """
    class PolicyNet(nn.Module):
        """
        policy(state) returns distribution over actions
        uses ScaledTanhTransformedGaussian as per Hafner
        """

        def __init__(self, input_dim, hidden_dim, output_dim=1, min=-1.0, max=1.0):
            super().__init__()
            self.mu = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dim, output_dim))
            self.scale = nn.Linear(input_dim, output_dim, bias=False)
            self.min = min
            self.max = max

        def forward(self, state):
            mu = self.mu(state)
            scale = torch.sigmoid(self.scale(state)) + 0.05
            return ScaledTanhTransformedGaussian(mu, scale, min=self.min, max=self.max)

    policy_net = PolicyNet(env.observation_space.shape[0],
                           16,
                           env.action_space.shape[0],
                           min=env.action_space.low[0],
                           max=env.action_space.high[0])

    """ policy to run on environment """
    def policy(state):
        state = torch.from_numpy(state).float()
        action = policy_net(state)
        a = action.rsample()
        return a.numpy()

    checkpoint.load('.', prefix='cartpole', policy_net=policy_net)

    while True:
        driver.episode(env, policy, render=True)