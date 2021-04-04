import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import max_pool2d, log_softmax
from torch.distributions import Categorical

import gym
import env
import env.wrappers as wrappers
from capture import LiveMonitor

import driver
from gymviz import Plot

import buffer as bf
from algos import ppo
from config import exists_and_not_none, ArgumentParser
import wandb
import wandb_utils
import checkpoint
import baselines.helper as helper
import numpy as np
import os
import capture
import random
import models.atari as models


if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='configuration switches')
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-r', '--run_id', type=int, default=-1)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--silent', action='store_true', default=False)

    """ reproducibility """
    parser.add_argument('--seed', type=int, default=None)

    """ main loop control """
    parser.add_argument('--max_steps', type=int, default=1000000)
    parser.add_argument('--test_steps', type=int, default=5000)
    parser.add_argument('--test_episodes', type=int, default=25)
    parser.add_argument('--capture_freq', type=int, default=50000)

    """ resume settings """
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)

    """ environment """
    parser.add_argument('--env_name', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--env_render', action='store_true', default=False)

    """ hyper-parameters """
    parser.add_argument('--optim_lr', type=float, default=1.3e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--exploration_noise', type=float, default=0.0565)

    config = parser.parse_args()

    """ check for device in environment variable """
    if 'DEVICE' in os.environ:
        config.device = os.environ['DEVICE']

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)
        #torch.backends.cudnn.benchmark = False
        #torch.use_deterministic_algorithms(True)

    wandb.init(project=f"ppo-a2c-{config.env_name}", config=config)

    """ environment """
    def make_env():
        env = gym.make(config.env_name)
        env = capture.VideoCapture(env, config.run_dir, freq=config.capture_freq)
        env = wrappers.EpisodicLifeEnv(env)
        if 'NOOP' in env.unwrapped.get_action_meanings():
            env = wrappers.NoopResetEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = wrappers.FireResetEnv(env)
        env = wrappers.ClipState2D(env, 0, 24, 210 - 24, 160)
        env = wrappers.WarpFrame(env)
        env = wrappers.ScaledFloatFrame(env)
        env = wrappers.Gradient(env)
        env = wrappers.ClipRewardEnv(env)
        env = wrappers.PenalizeOneIfDone(env)

        if config.seed is not None:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
        return env

    """ training env with replay buffer """
    train_env, train_buffer = bf.wrap(make_env())
    train_env = wandb_utils.LogRewards(train_env)
    if not config.silent:
        train_env = Plot(train_env, episodes_per_point=20, title=f'Train ppo-a2c-{config.env_name}-{config.run_id}',
                         refresh_cooldown=5.0)

    """ test env """
    test_env = make_env()
    if not config.silent:
        test_env = Plot(test_env, episodes_per_point=config.test_episodes, title=f'Test ppo-a2c-{config.env_name}-{config.run_id}',
                        refresh_cooldown=5.0)
    evaluator = helper.Evaluator(test_env)

    actions = train_env.action_space.n

    """ network """
    a2c_net = models.A2CNet(
        hidden_dims=config.hidden_dim,
        actions=actions,
        exploration_noise=config.exploration_noise)
    optim = torch.optim.Adam(a2c_net.parameters(), lr=config.optim_lr, weight_decay=0.0)

    a2c_net = ppo.PPOWrapModel(a2c_net).to(config.device)

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        checkpoint.load(config.load, prefix='best', a2c_net=a2c_net, optim=optim)

    """ policy to run on environment """
    def policy(state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(config.device)
        value, action = a2c_net(state)
        a = action.sample()
        assert torch.isnan(a) == False
        return a.cpu().numpy()

    """ demo  """
    evaluator.demo(config.demo, policy)

    """ main loop """
    steps = 0
    best_mean_return = -999999
    tests_run = 0

    """ sample from batch_size transitions from the replay buffer """
    dl = DataLoader(train_buffer, batch_size=config.batch_size)

    for total_steps, _ in enumerate(driver.step_environment(train_env, policy)):
        steps += 1
        if total_steps > config.max_steps:
            break

        """ train online after batch steps saved"""
        if steps < config.batch_size:
            continue
        else:
            ppo.train_a2c(dl, a2c_net, optim, discount=config.discount, batch_size=config.batch_size, device=config.device)
            train_buffer.clear()
            steps = 0

        """ test  """
        if total_steps > config.test_steps * tests_run and total_steps != 0:
            tests_run += 1
            evaluator.evaluate(policy, config.run_dir, {'a2c_net': a2c_net, 'optim': optim},
                               sample_n=config.test_episodes, capture=True)