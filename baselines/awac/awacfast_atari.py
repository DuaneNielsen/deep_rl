import torch
import torch.nn as nn

import gym
import env
import env.wrappers as wrappers
import capture

import driver
from gymviz import Plot

import buffer as bf
from algos import awac
from torch.distributions import Categorical
from config import exists_and_not_none, ArgumentParser
import wandb
import wandb_utils
import checkpoint
import baselines.helper as helper
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader, RandomSampler
from gym.wrappers.transform_reward import TransformReward
from time import time
from statistics import mean, stdev
from random import randint, sample
import os
import random

def rescale_reward(reward):
    return reward * config.env_reward_scale - config.env_reward_bias



if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='configuration switches')
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-d', '--device', type=str)
    parser.add_argument('-r', '--run_id', type=int, default=-1)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)

    """ reproducibility """
    parser.add_argument('--seed', type=int, default=None)

    """ main loop control """
    parser.add_argument('--max_steps', type=int, default=150000)
    parser.add_argument('--test_steps', type=int, default=5000)
    parser.add_argument('--test_samples', type=int, default=5)
    parser.add_argument('--test_capture', action='store_true', default=True)
    parser.add_argument('--capture_freq', type=int, default=5000)

    """ resume settings """
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)
    parser.add_argument('--load_buffer', type=str, default='buffer.pkl')

    """ environment """
    parser.add_argument('--env_name', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--env_render', action='store_true', default=False)
    parser.add_argument('--env_reward_scale', type=float, default=1.0)
    parser.add_argument('--env_reward_bias', type=float, default=0.0)
    parser.add_argument('--env_timelimit', type=int, default=3000)

    """ hyper-parameters """
    parser.add_argument('--optim_lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--exploration_noise', type=float, default=0.05)
    parser.add_argument('--lam', type=float, default=1.0)

    """ experimental parameters """
    parser.add_argument('--buffer_steps', type=int)
    parser.add_argument('--buffer_capacity', type=int)

    config = parser.parse_args()

    if config.buffer_capacity is None:
        config.buffer_capacity = config.max_steps

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)
        random.seed(config.seed)

    if 'DEVICE' in os.environ:
        config.device = os.environ['DEVICE']

    wandb.init(project=f"awac-{config.env_name}", config=config)

    """ environment """
    def make_env():
        env = gym.make(config.env_name)
        env = wrappers.TimeLimit(env.unwrapped, max_episode_steps=config.env_timelimit)
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
        env = TransformReward(env, rescale_reward)

        if config.seed is not None:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
        return env

    """ training env with replay buffer """
    train_env, train_buffer = bf.wrap(make_env())

    class FastOfflineDataset:
        def __init__(self, load_buff, capacity, device='cpu', length=None, rescale_reward=1.0):

            if length is None:
                length = len(load_buff)

            self.device = device

            s, a, s_p, r, d = load_buff[0]

            self.length = min(len(load_buff), length)
            self.capacity = max(length, capacity)
            self.state = torch.empty(self.capacity, *s.shape, dtype=torch.float32, device=device)
            self.action = torch.empty(self.capacity, 1, dtype=torch.long, device=device)
            self.state_p = torch.empty(self.capacity, *s_p.shape, dtype=torch.float32, device=device)
            self.reward = torch.empty(self.capacity, 1, dtype=torch.float32, device=device)
            self.done = torch.empty(self.capacity, 1, dtype=torch.float32, device=device)

            for i, sampled in enumerate(sample(range(len(load_buff)), self.length)):
                self[i] = load_buff[sampled]

            self.reward = self.reward * rescale_reward

        def __len__(self):
            return self.length

        def __getitem__(self, i):
            return self.state[i], self.action[i], self.state_p[i], self.reward[i], self.done[i]

        def __setitem__(self, i, transition):
            s, a, s_p, r, d = transition
            self.state[i] = torch.from_numpy(s).type(torch.float32)
            self.action[i] = a
            self.state_p[i] = torch.from_numpy(s_p).type(torch.float32)
            self.reward[i] = r
            self.done[i] = 0.0 if d else 1.0

        def append(self, transition):
            # if capacity in the buffer, append, else overwrite at random
            if self.length < self.capacity:
                i = self.length
                self.length += 1
            else:
                i = randint(0, self.length-1)
            self[i] = transition


    load_buff = bf.load(config.load_buffer)

    tds = FastOfflineDataset(load_buff, length=config.buffer_steps, capacity=config.buffer_capacity, rescale_reward=config.env_reward_scale)

    train_env = wandb_utils.LogRewards(train_env)

    if not config.silent:
        train_env = Plot(train_env, episodes_per_point=1, title=f'Train awac-{config.env_name}')

    """ test env """
    test_env = make_env()
    if not config.silent:
        test_env = Plot(test_env, episodes_per_point=1, title=f'Test awac-{config.env_name}')
    evaluator = helper.Evaluator(test_env)


    class AtariVision(nn.Module):
        def __init__(self, feature_size=512):
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
                nn.Conv2d(256, feature_size, kernel_size=3, stride=1, padding=1),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))

        def forward(self, state):
            l1 = self.conv1(state.permute(0, 3, 1, 2))
            l2 = self.conv2(l1)
            l3 = self.conv3(l2)
            l4 = self.conv4(l3)
            l5 = self.conv5(l4)
            l6 = self.conv6(l5)
            return l6.flatten(start_dim=1)

    class Q(nn.Module):
        def __init__(self, feature_size, hidden_dims, actions):
            super().__init__()
            self.q = nn.Sequential(
                AtariVision(feature_size),
                nn.Linear(feature_size, hidden_dims), nn.SELU(inplace=True),
                nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                nn.Linear(hidden_dims, actions))

        def forward(self, state):
            return self.q(state)

    class Policy(nn.Module):
        def __init__(self, feature_size, hidden_dims, actions, exploration_noise):
            super().__init__()
            self.exploration_noise = exploration_noise
            self.actions = actions
            self.policy = nn.Sequential(
                AtariVision(feature_size),
                nn.Linear(feature_size, hidden_dims), nn.SELU(inplace=True),
                nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                nn.Linear(hidden_dims, actions))

        def forward(self, state):
            action = log_softmax(self.policy(state), dim=1)
            action = torch.log((1 - self.exploration_noise) * torch.exp(action) +
                               self.exploration_noise * torch.ones_like(action) / self.actions)
            a_dist = Categorical(logits=action)
            return a_dist

    class A2CNet(nn.Module):
        def __init__(self, feature_size, hidden_dims, actions, exploration_noise):
            """

            Args:
                feature_size: features vector size output by the convolution layers
                hidden_dims: hidden dim size of FC mlp
                actions: number of discrete actions
                exploration_noise:
            """
            super().__init__()
            self.actions = actions
            self.exploration_noise = exploration_noise

            self.q = Q(feature_size, hidden_dims, actions)
            self.policy = Policy(feature_size, hidden_dims, actions, exploration_noise)

        def forward(self, state):
            value = self.q(state)
            a_dist = self.policy(state)
            return value, a_dist


    awac_net = A2CNet(
        feature_size=512,
        actions=test_env.action_space.n,
        hidden_dims=config.hidden_dim,
        exploration_noise=config.exploration_noise).to(config.device)

    q_optim = torch.optim.Adam(awac_net.q.parameters(), lr=config.optim_lr)
    policy_optim = torch.optim.Adam(awac_net.policy.parameters(), lr=config.optim_lr)

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        checkpoint.load(config.load, prefix='best', awac_net=awac_net, q_optim=q_optim, policy_optim=policy_optim)

    """ policy to run on environment """
    def policy(state):
        state = torch.from_numpy(state).unsqueeze(0)
        state = state.to(config.device)
        value, action = awac_net(state)
        a = action.sample()
        assert torch.isnan(a) == False
        return a.item()

    """ demo  """
    evaluator.demo(config.demo, policy)

    """ main loop """
    steps = 0
    best_mean_return = -999999
    tests_run = 0

    offline_steps = len(tds)

    num_workers = 0 if config.debug else 2

    dl = DataLoader(tds, batch_size=config.batch_size, sampler=RandomSampler(tds, replacement=False),
                    num_workers=num_workers)

    wandb.run.summary['offline_steps'] = offline_steps
    on_policy = False
    print(f'OFF POLICY FOR {len(tds)} steps')
    train_time = []
    frame_time = []

    start = time()

    for total_steps, (s, a, s_p, r, d, _) in enumerate(driver.step_environment(train_env, policy)):

        if total_steps > config.max_steps:
            break

        get_frame = time()

        if total_steps > offline_steps:
            tds.append((s, a, s_p, r, d))
            if not on_policy:
                print('on policy NOW')
                on_policy = True

        """ train offline after batch steps saved"""
        if len(tds) < config.batch_size:
            continue
        else:
            awac.train_discrete(dl, awac_net, q_optim, policy_optim, lam=config.lam,
                            device=config.device, debug=config.debug, measure_kl=True)

        train = time()

        """ test  """
        if total_steps > config.test_steps * (tests_run + 1):
            tests_run += 1
            evaluator.evaluate(policy, config.run_dir, sample_n=config.test_samples, capture=config.test_capture,
                               params={'awac_net': awac_net, 'q_optim': q_optim, 'policy_optim': policy_optim})

        train_time += [train - get_frame]
        frame_time += [get_frame - start]

        start = time()

        if total_steps % 100 == 0:
            print(f'train_time: {mean(train_time)}, frame_time: {mean(frame_time)}')
            train_time, frame_time = [], []