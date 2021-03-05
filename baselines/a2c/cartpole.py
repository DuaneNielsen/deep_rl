import torch
import torch.nn as nn

import gym
import env

import driver
from gymviz import Plot

import buffer as bf
from algos import a2c
from distributions import ScaledTanhTransformedGaussian


class RescaleReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return reward / 200.0


if __name__ == '__main__':

    """ configuration """
    batch_size = 16  # number of trajectories per training iteration
    discount = 0.95

    """ environment """
    env = gym.make('CartPoleContinuous-v1')
    env = RescaleReward(env)  # rescale reward to reduce gradient size

    """ replay buffer """
    env, buffer = bf.wrap(env)
    env = Plot(env, blocksize=batch_size)

    class VNet(nn.Module):
        """
        policy(state) returns distribution over actions
        uses ScaledTanhTransformedGaussian as per Hafner
        """
        def __init__(self):
            super().__init__()
            self.mu = nn.Sequential(nn.Linear(4, 12), nn.SELU(inplace=True),
                                    nn.Linear(12, 12), nn.ELU(inplace=True),
                                    nn.Linear(12, 1))

        def forward(self, state):
            return self.mu(state)

    class PolicyNet(nn.Module):
        """
        policy(state) returns distribution over actions
        uses ScaledTanhTransformedGaussian as per Hafner
        """
        def __init__(self, min=-1.0, max=1.0):
            super().__init__()
            self.mu = nn.Sequential(nn.Linear(4, 12), nn.SELU(inplace=True),
                                    nn.Linear(12, 12), nn.ELU(inplace=True),
                                    nn.Linear(12, 1))
            self.scale = nn.Linear(4, 1, bias=False)
            self.min = min
            self.max = max

        def forward(self, state):
            mu = self.mu(state)
            scale = torch.sigmoid(self.scale(state)) + 0.1
            return ScaledTanhTransformedGaussian(mu, scale, min=self.min, max=self.max)

    v_net = VNet()
    v_optim = torch.optim.Adam(v_net.parameters(), lr=1e-4)
    policy_net = PolicyNet()
    policy_optim = torch.optim.Adam(policy_net.parameters(), lr=1e-4)

    """ policy to run on environment """
    def policy(state):
        state = torch.from_numpy(state).float()
        action = policy_net(state)
        a = action.rsample().numpy()
        return a

    """ sample """
    for epoch in range(8000):
        for ep in range(batch_size):
            driver.episode(env, policy)

        a2c.train(buffer, v_net, v_optim, policy_net, policy_optim)