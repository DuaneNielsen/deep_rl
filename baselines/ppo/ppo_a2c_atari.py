import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import max_pool2d, log_softmax
from torch.distributions import Categorical

import gym
import env
import env.wrappers as wrappers
#from capture import LiveMonitor

import driver
from gymviz import Plot

import buffer as bf
from algos import ppo
from config import exists_and_not_none, ArgumentParser, EvalAction
import wandb
import wandb_utils
import checkpoint
import baselines.helper as helper
import numpy as np
import os
# import capture
import random
import models.atari as models


if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='configuration switches')
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-r', '--run_id', type=int, default=-1)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--precision', type=str, action=EvalAction, default=torch.float32)

    """ reproducibility """
    parser.add_argument('--seed', type=int, default=None)

    """ main loop control """
    parser.add_argument('--max_steps', type=int, default=1000000)
    parser.add_argument('--test_steps', type=int, default=5000)
    parser.add_argument('--test_episodes', type=int, default=25)

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

    wandb.init(project=f"ppo-a2c-{config.env_name}", config=config)

    """ environment """
    def make_env():
        env = gym.make(config.env_name)
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
    train_env = make_env()
    train_env = wandb_utils.LogRewards(train_env)
    if config.debug:
        train_env = Plot(train_env, episodes_per_point=20, title=f'Train ppo-a2c-{config.env_name}-{config.run_id}',
                         refresh_cooldown=5.0)

    """ test env """
    test_env = make_env()
    if config.debug:
        test_env = Plot(test_env, episodes_per_point=config.test_episodes, title=f'Test ppo-a2c-{config.env_name}-{config.run_id}',
                        refresh_cooldown=5.0)

    """ network """
    a2c_net = models.A2CNet(
        hidden_dims=config.hidden_dim,
        actions=train_env.action_space.n,
        exploration_noise=config.exploration_noise)
    optim = torch.optim.Adam(a2c_net.parameters(), lr=config.optim_lr, weight_decay=0.0)

    a2c_net = ppo.PPOWrapModel(a2c_net).to(config.device)

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        checkpoint.load(config.load, prefix='best', a2c_net=a2c_net, optim=optim)

    """ policy to run on environment """
    def policy(state):
        with torch.no_grad():
            state = torch.from_numpy(state).type(config.precision).unsqueeze(0).to(config.device)
            value, action = a2c_net(state)
            a = action.sample()
            assert torch.isnan(a) == False
            return a.item()

    """ demo  """
    helper.demo(config.demo, env, policy)

    """ training loop """
    buffer = []
    dl = DataLoader(buffer, batch_size=config.batch_size)
    evaluator = helper.Evaluator()

    for global_step, (s, a, s_p, r, d, i) in enumerate(bf.step_environment(train_env, policy)):

        buffer.append((s, a, s_p, r, d))

        if len(buffer) == config.batch_size:
            ppo.train_a2c(dl, a2c_net, optim, discount=config.discount,
                          device=config.device, precision=config.precision)
            buffer.clear()

        """ test  """
        if evaluator.evaluate_now(global_step, config.test_steps):
            evaluator.evaluate(test_env, policy, config.run_dir, params={'a2c_net': a2c_net, 'optim': optim},
                               sample_n=config.test_episodes, capture=True)

        if global_step > config.max_steps:
            break
