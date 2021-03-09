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
from config import ArgumentParser, exists_and_not_none
import checkpoint


class RescaleReward(gym.RewardWrapper):
    def __init__(self, env, multiple=1.0, bias=0.0):
        super().__init__(env)
        self.multiple = multiple
        self.bias = bias

    def reward(self, reward):
        return reward * self.multiple + self.bias


if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='configuration switches')
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-d', '--device', type=str)
    parser.add_argument('-r', '--run_id', type=int, default=-1)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--silent', action='store_true', default=False)

    """ reproducibility """
    parser.add_argument('--seed', type=int, default=None)

    """ main loop control """
    parser.add_argument('--max_steps', type=int, default=2000000)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--test_steps', type=int, default=100000)
    parser.add_argument('--test_episodes', type=int, default=10)

    """ resume settings """
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)

    """ environment """
    parser.add_argument('--env_name', type=str, default='CartPoleContinuous-v1')
    parser.add_argument('--env_render', action='store_true', default=False)
    parser.add_argument('--env_reward_multiple', type=float, default=1.0)
    parser.add_argument('--env_reward_bias', type=float, default=0.0)
    parser.add_argument('--env_action_max', type=float, default=1.0)
    parser.add_argument('--env_action_min', type=float, default=-1.0)

    """ hyper-parameters """
    parser.add_argument('--optim_class', type=str)
    parser.add_argument('--optim_lr', type=float, default=1e-4)
    parser.add_argument('--scheduler-class', type=str)
    parser.add_argument('--episodes_per_batch', type=int, default=16)
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--min_variance', type=float, default=0.05)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--clip_max', type=float, default=-0.1)
    parser.add_argument('--clip_min', type=float, default=-2.0)

    config = parser.parse_args()

    wandb.init(project=f"reinforce-{config.env_name}", config=config)

    """ environment """
    def make_env():
        env = gym.make(config.env_name)
        # rescale reward to reduce gradient size
        env = RescaleReward(env, multiple=config.env_reward_multiple, bias=config.env_reward_bias)
        assert len(env.observation_space.shape) == 1, "Only 1D continuous observation space is supported"
        assert len(env.action_space.shape) == 1, "Only 1D continuous action space is supported"
        return env

    env = make_env()

    """ replay buffer """
    train_env, buffer = bf.wrap(env)
    buffer.enrich(bf.DiscountedReturns(discount=config.discount))
    if not config.silent:
        train_env = Plot(train_env, episodes_per_point=config.episodes_per_batch)
    train_env = wandb_utils.LogRewards(train_env)

    test_env = make_env()
    if not config.silent:
        test_env = Plot(make_env())

    """ random seed """
    if exists_and_not_none(config, 'seed'):
        torch.manual_seed(config.seed)
        train_env.seed(config.seed)
        train_env.action_space.seed(config.seed)
        test_env.seed(config.seed)
        test_env.action_space.seed(config.seed)

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
            scale = torch.sigmoid(self.scale(state)) + config.min_variance
            return ScaledTanhTransformedGaussian(mu, scale, min=self.min, max=self.max)


    policy_net = PolicyNet(env.observation_space.shape[0],
                           config.hidden_dim,
                           env.action_space.shape[0],
                           max=config.env_action_max,
                           min=config.env_action_min)

    optim = torch.optim.Adam(policy_net.parameters(), lr=config.optim_lr)

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        checkpoint.load(config.load, prefix='best', policy_net=policy_net, optim=optim)

    wandb.watch(policy_net)

    """ policy to run on environment """
    def policy(state):
        state = torch.from_numpy(state).float()
        action = policy_net(state)
        a = action.rsample()
        if torch.isnan(a):
            wandb.run.summary["FAIL"] = str(action.mu) + " " + str(action.scale)
        assert torch.isnan(a) == False
        return a.numpy()

    """ demo  """
    if config.demo:
        while True:
            driver.episode(test_env, policy, render=True)
            buffer.clear()

    best_mean_return = -999999
    total_steps = 0
    tests_run = 0

    """ main loop """
    for epoch in range(config.epochs):
        for ep in range(config.episodes_per_batch):
            driver.episode(train_env, policy)
        total_steps += len(buffer)

        """ train """
        reinforce.train(buffer, policy_net, optim, clip_min=config.clip_min, clip_max=config.clip_max)

        """ test  """
        if total_steps > config.test_steps * tests_run:
            tests_run += 1

            mean_return, stdev_return = checkpoint.sample_policy_returns(test_env, policy, config.test_episodes,
                                                                         render=config.env_render)

            wandb.run.summary["last_mean_return"] = mean_return
            wandb.run.summary["last_stdev_return"] = stdev_return

            # checkpoint policy if mean return is better
            if mean_return > best_mean_return:
                best_mean_return = mean_return
                wandb.run.summary["best_mean_return"] = best_mean_return
                wandb.run.summary["best_stdev_return"] = stdev_return
                checkpoint.save(config.run_dir, 'best', policy_net=policy_net, optim=optim)

        if total_steps > config.max_steps:
            break
