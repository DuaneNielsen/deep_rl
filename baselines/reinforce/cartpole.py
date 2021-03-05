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


class RescaleReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return reward / 200.0


if __name__ == '__main__':

    """ configuration """
    config = dict(
        episodes_per_batch=16,  # number of trajectories per training iteration
        discount=0.95,
        min_variance=0.05,
        env='CartPoleContinuous-v1'
    )

    wandb.init(project="reinforce-cartpole", config=config)
    config = wandb.config

    """ environment """
    env = gym.make(config.env)
    env = RescaleReward(env)  # rescale reward to reduce gradient size

    """ replay buffer """
    env, buffer = bf.wrap(env)
    buffer.attach_enrichment(bf.DiscountedReturns(discount=config.discount))
    env = Plot(env, episodes_per_point=config.episodes_per_batch)
    env = wandb_utils.LogRewards(env)

    RANDOM_SEED = 0
    torch.manual_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

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
            scale = torch.sigmoid(self.scale(state)) + config.min_variance
            return ScaledTanhTransformedGaussian(mu, scale, min=self.min, max=self.max)


    policy_net = PolicyNet()
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
            driver.episode(env, policy)

        reinforce.train(buffer, policy_net, optim)

        """ test loop goes here """
        if epoch % 100:
            pass