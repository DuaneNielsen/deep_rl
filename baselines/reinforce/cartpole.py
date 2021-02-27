import torch
import torch.nn as nn

import gym


import buffer as bf
from algos import reinforce
from distributions import ScaledTanhTransformedGaussian
from env.continuous_cartpole import ContinuousCartPoleEnv


class RescaleReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return reward / 200.0


if __name__ == '__main__':

    batch_size = 128

    env = ContinuousCartPoleEnv()
    env = RescaleReward(env)
    env = bf.SubjectWrapper(env)
    buffer = bf.ReplayBuffer()
    buffer.attach_enrichment(bf.DiscountedReturns(discount=0.95))
    env.attach_observer("replay_buffer", buffer)
    panel = bf.Plot(blocksize=batch_size)
    env.attach_observer('panel', panel)

    class PolicyNet(nn.Module):
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
            scale = torch.sigmoid(self.scale(state)) + 0.05
            return ScaledTanhTransformedGaussian(mu, scale, min=self.min, max=self.max)

    policy_net = PolicyNet()
    optim = torch.optim.Adam(policy_net.parameters(), lr=1e-4)

    def policy(state):
        state = torch.from_numpy(state).float()
        action = policy_net(state)
        a = action.rsample().numpy()
        return a

    for epoch in range(2048):
        for ep in range(128):
            bf.episode(env, policy)

        reinforce.train(buffer, policy_net, optim)
        buffer.clear()

    assert False