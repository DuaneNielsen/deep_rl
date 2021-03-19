import torch
import torch.nn as nn

import gym
import env
import env.wrappers as wrappers
from capture import LiveMonitor

import driver
from gymviz import Plot

import buffer as bf
from algos import ppo
from torch.nn.functional import max_pool2d, softmax
from torch.distributions import Categorical
from config import exists_and_not_none, ArgumentParser
import wandb
import wandb_utils
import checkpoint
import baselines.helper as helper
import numpy as np
import os
import capture
import random

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
    parser.add_argument('--max_steps', type=int, default=100000000)
    parser.add_argument('--test_steps', type=int, default=3000000)
    parser.add_argument('--test_episodes', type=int, default=10)
    parser.add_argument('--capture_freq', type=int, default=5000)

    """ resume settings """
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)

    """ environment """
    parser.add_argument('--env_name', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--env_render', action='store_true', default=False)

    """ hyper-parameters """
    parser.add_argument('--optim_lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--exploration_noise', type=float, default=0.3)

    config = parser.parse_args()

    """ check for device in environment variable """
    if 'DEVICE' in os.environ:
        config.device = os.environ['DEVICE']

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)
        #torch.backends.cudnn.benchmark = False
        #torch.use_deterministic_algorithms(True)

    wandb.init(project=f"ppp-a2c-{config.env_name}", config=config)

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
        #env = RemapActions(env, remap=np.array([1, 2, 3]))

        #env = LiveMonitor(env)

        if config.seed is not None:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
        return env

    """ training env with replay buffer """
    train_env, train_buffer = bf.wrap(make_env())
    train_env = wandb_utils.LogRewards(train_env)
    if not config.silent:
        train_env = Plot(train_env, episodes_per_point=20, title=f'Train ppo-a2c-{config.env_name}-{config.run_id}',
                         history_length=200, refresh_cooldown=5.0)

    """ test env """
    test_env = make_env()
    if not config.silent:
        test_env = Plot(test_env, episodes_per_point=config.test_episodes, title=f'Test ppo-a2c-{config.env_name}-{config.run_id}',
                        history_length=200, refresh_cooldown=5.0)
    evaluator = helper.Evaluator(test_env)

    actions = train_env.action_space.n

    """ network """
    class A2CNet(nn.Module):
        """
        policy(state) returns distribution over actions
        uses ScaledTanhTransformedGaussian as per Hafner
        """
        def __init__(self, input_dims, hidden_dims, actions):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.conv3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.conv4 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.conv5 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.conv6 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))


            self.value = nn.Sequential(nn.Linear(512, hidden_dims), nn.SELU(inplace=True),
                                     nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                                     nn.Linear(hidden_dims, 1))

            self.action = nn.Sequential(nn.Linear(512, hidden_dims), nn.SELU(inplace=True),
                                     nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                                     nn.Linear(hidden_dims, actions))

        def forward(self, state):
            l1 = self.conv1(state.permute(0, 3, 1, 2))
            l2 = self.conv2(l1)
            l3 = self.conv3(l2)
            l4 = self.conv4(l3)
            l5 = self.conv5(l4)
            l6 = self.conv6(l5)

            value = self.value(l6.flatten(start_dim=1))
            action = softmax(self.action(l6.flatten(start_dim=1)), dim=1)
            noise = softmax(torch.ones_like(action), dim=1)
            action = (1 - config.exploration_noise) * action + config.exploration_noise * noise
            a_dist = Categorical(action)
            return value, a_dist

    a2c_net = A2CNet(
        input_dims=test_env.observation_space.shape[0],
        hidden_dims=config.hidden_dim,
        actions=actions)
    optim = torch.optim.Adam(a2c_net.parameters(), lr=config.optim_lr, weight_decay=0.0)

    a2c_net = ppo.PPOWrapModel(a2c_net).to(config.device)

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        checkpoint.load(config.load, prefix='best', a2c_net=a2c_net, optim=optim)
        a2c_net.backup()  # old parameters don't get stored, so use the new one

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

    for total_steps, _ in enumerate(driver.step_environment(train_env, policy)):
        steps += 1
        if total_steps > config.max_steps:
            break

        """ train online after batch steps saved"""
        if steps < config.batch_size:
            continue
        else:
            ppo.train_a2c(train_buffer, a2c_net, optim, discount=config.discount, batch_size=config.batch_size, device=config.device)
            steps = 0

        """ test  """
        if total_steps > config.test_steps * tests_run and total_steps != 0:
            tests_run += 1
            evaluator.evaluate(policy, config.run_dir, {'a2c_net': a2c_net, 'optim': optim},
                               sample_n=config.test_episodes)