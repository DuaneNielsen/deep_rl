import torch
import torch.nn as nn

import gym
import env

import driver
from gymviz import Plot

import buffer as bf
from algos import a2c
from distributions import ScaledTanhTransformedGaussian
from config import exists_and_not_none


class RescaleReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return reward / 100.0


if __name__ == '__main__':

    """ configuration """
    batch_size = 8
    discount = 0.99
    max_steps = 20000000
    min_action = - 1.0
    max_action = 1.0

    class Config:
        def __init__(self):
            self.seed = 0
    config = Config()


    """ environment """
    env = gym.make('CartPoleContinuous-v1')

    """ replay buffer """
    env, buffer = bf.wrap(env)
    env = Plot(env, episodes_per_point=5)

    """ random seed """
    if exists_and_not_none(config, 'seed'):
        torch.manual_seed(config.seed)
        env.seed(config.seed)
        env.action_space.seed(config.seed)

    class A2CNet(nn.Module):
        """
        policy(state) returns distribution over actions
        uses ScaledTanhTransformedGaussian as per Hafner
        """
        def __init__(self, min, max):
            super().__init__()
            self.min = min
            self.max = max
            self.mu = nn.Sequential(nn.Linear(4, 32), nn.SELU(inplace=True),
                                    nn.Linear(32, 32), nn.SELU(inplace=True),
                                    nn.Linear(32, 2))
            self.scale = nn.Linear(4, 1, bias=False)

        def forward(self, state):
            output = self.mu(state)
            value = output[..., 0:1]
            mu = output[..., 1:2]
            scale = torch.sigmoid(self.scale(state)) + 0.01
            a_dist = ScaledTanhTransformedGaussian(mu, scale, min=self.min, max=self.max)
            return value, a_dist

    a2c_net = A2CNet(min=-1.0, max=1.0)
    optim = torch.optim.Adam(a2c_net.parameters(), lr=1e-4)

    """ policy to run on environment """
    def policy(state):
        state = torch.from_numpy(state).float()
        value, action = a2c_net(state)
        a = action.rsample()
        assert torch.isnan(a) == False
        return a.numpy()

    """ main loop """
    steps = 0
    for total_steps, _ in enumerate(driver.step_environment(env, policy)):
        steps += 1
        if total_steps > max_steps:
            break

        """ train online after batch steps saved"""
        if steps < batch_size:
            continue
        else:
            a2c.train(buffer, a2c_net, optim, batch_size=batch_size)
            steps = 0