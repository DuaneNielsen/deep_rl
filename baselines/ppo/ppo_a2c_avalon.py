import torch

import algos.utils
import avalonsim
from models.mlp import MLP, ValueHead, ActionHead
from torch.utils.data import DataLoader

import gym
import env
from copy import deepcopy
import env.wrappers as wrappers
# from capture import LiveMonitor

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
from avalonsim.wrappers import RandomEnemyWrapper
from time import time

if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='configuration switches')
    parser.add_argument('-c', '--config', type=str, default='avalon.yaml')
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
    parser.add_argument('--env_name', type=str, default='Avalon-v1')
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
        algos.utils.seed_all(config.seed)

    wandb.init(project=f"ppo-a2c-{config.env_name}", config=config)

    """ environment """


    def make_env():
        env = gym.make(config.env_name)
        env = RandomEnemyWrapper(env)

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
        test_env = Plot(test_env, episodes_per_point=config.test_episodes,
                        title=f'Test ppo-a2c-{config.env_name}-{config.run_id}',
                        refresh_cooldown=5.0)

    """ network """
    value_net = MLP(
        in_features=train_env.observation_space.shape[0],
        hidden_dims=config.hidden_dim,
        head=ValueHead(
            hidden_dims=config.hidden_dim
        )
    ).to(config.device)

    value_optim = torch.optim.Adam(value_net.parameters(), lr=config.optim_lr, weight_decay=0.0)

    policy_net = MLP(
        in_features=train_env.observation_space.shape[0],
        hidden_dims=config.hidden_dim,
        head=ActionHead(
            hidden_dims=config.hidden_dim,
            actions=train_env.action_space.n,
            exploration_noise=config.exploration_noise)
    )

    policy_optim = torch.optim.Adam(policy_net.parameters(), lr=config.optim_lr, weight_decay=0.0)

    policy_net = ppo.PPOWrapModel(policy_net).to(config.device)

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        checkpoint.load(config.load, prefix='best', policy_net=policy_net, policy_optim=policy_optim,
                        value_net=value_net, value_optim=value_optim)

    """ policy to run on environment """
    def policy(state):
        with torch.no_grad():
            state = torch.from_numpy(state).type(config.precision).unsqueeze(0).to(config.device)
            value, action = value_net(state), policy_net(state)
            a = action.sample()
            assert torch.isnan(a) == False
            return a.item()


    """ demo  """
    helper.demo(config.demo, env, policy)

    """ training loop """
    buffer = []
    dl = DataLoader(buffer, batch_size=config.batch_size)
    evaluator = helper.Evaluator()

    start_t = time()
    env_time = 0
    train_time = 0.

    for global_step, (s, a, s_p, r, d, i) in enumerate(bf.step_environment(train_env, policy)):
        env_t = time()
        env_time = (env_t - start_t) * 0.01 + 0.99 * env_time

        buffer.append((s, a, s_p, r, d))

        if len(buffer) == config.batch_size:
            ppo.train_a2c_stable(dl, value_net, value_optim, policy_net, policy_optim,
                                 discount=config.discount, device=config.device, precision=config.precision)
            buffer.clear()

            train_t = time()
            train_time = (train_t - env_t) * 0.01 + 0.99 * train_time

        start_t = time()

        if global_step % 1000 == 0:
            print(env_time * config.batch_size, train_time)

        """ test  """
        if evaluator.evaluate_now(global_step, config.test_steps):
            evaluator.evaluate(test_env, policy, config.run_dir,
                               params={
                                   'policy_net': policy_net, 'policy_optim': policy_optim,
                                   'value_net': value_net, 'value_optim': value_optim
                               },
                               sample_n=config.test_episodes, capture=False)

        if global_step > config.max_steps:
            break
