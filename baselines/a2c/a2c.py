import torch
import torch.nn as nn

import gym
import env

import driver
from gymviz import Plot

import buffer as bf
from algos import a2c
from distributions import ScaledTanhTransformedGaussian
from config import exists_and_not_none, ArgumentParser
import wandb
import wandb_utils
import checkpoint


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
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--test_steps', type=int, default=30000)
    parser.add_argument('--test_episodes', type=int, default=10)

    """ resume settings """
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)

    """ environment """
    parser.add_argument('--env_name', type=str, default='CartPoleContinuous-v1')
    parser.add_argument('--env_render', action='store_true', default=False)

    """ hyper-parameters """
    parser.add_argument('--optim_lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=16)

    config = parser.parse_args()

    wandb.init(project=f"a2c-{config.env_name}", config=config)

    """ environment """
    def make_env():
        env = gym.make(config.env_name)
        return env

    """ replay buffer """
    train_env = make_env()
    train_env, buffer = bf.wrap(train_env)
    if not config.silent:
        train_env = Plot(train_env, episodes_per_point=5, title=f"Train a2c-{config.env_name}")
    train_env = wandb_utils.LogRewards(train_env)

    test_env = make_env()
    if not config.silent:
        test_env = Plot(make_env(), title=f"Test a2c-{config.env_name}")

    """ random seed """
    if exists_and_not_none(config, 'seed'):
        torch.manual_seed(config.seed)
        train_env.seed(config.seed)
        train_env.action_space.seed(config.seed)
        test_env.seed(config.seed)
        test_env.action_space.seed(config.seed)

    """ network """
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

    a2c_net = A2CNet(min=test_env.action_space.low[0],
                     max=test_env.action_space.high[0])
    optim = torch.optim.Adam(a2c_net.parameters(), lr=config.optim_lr)

    """ policy to run on environment """
    def policy(state):
        state = torch.from_numpy(state).float()
        value, action = a2c_net(state)
        a = action.rsample()
        assert torch.isnan(a) == False
        return a.numpy()

    """ main loop """
    steps = 0
    best_mean_return = -999999
    tests_run = 0

    for total_steps, _ in enumerate(driver.step_environment(train_env, policy)):
        steps += 1
        if total_steps > config.max_steps:
            break

        """ train online after batch steps saved"""
        if steps < config.batch_size:
            continue
        else:
            a2c.train(buffer, a2c_net, optim, batch_size=config.batch_size)
            steps = 0

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
                checkpoint.save(config.run_dir, 'best', a2c_net=a2c_net, optim=optim)