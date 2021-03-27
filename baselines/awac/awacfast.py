import torch
import torch.nn as nn

import gym
import env

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
import pickle
from gym.wrappers.transform_reward import TransformReward
from time import time
from statistics import mean, stdev
from torch.utils.data import TensorDataset
from collections import namedtuple
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

    """ reproducibility """
    parser.add_argument('--seed', type=int, default=None)

    """ main loop control """
    parser.add_argument('--max_steps', type=int, default=150000)
    parser.add_argument('--test_steps', type=int, default=2000)
    parser.add_argument('--test_samples', type=int, default=5)
    parser.add_argument('--test_capture', action='store_true', default=False)

    """ resume settings """
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)
    parser.add_argument('--load_buffer', type=str, default='buffer.pkl')

    """ environment """
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--env_render', action='store_true', default=False)
    parser.add_argument('--env_reward_scale', type=float, default=1.0)
    parser.add_argument('--env_reward_bias', type=float, default=0.0)

    """ hyper-parameters """
    parser.add_argument('--optim_lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=16)

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
        env = TransformReward(env, rescale_reward)
        if config.seed is not None:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
        return env

    """ training env with replay buffer """
    train_env, train_buffer = bf.wrap(make_env())


    #tds = TensorDataset(state, action, state_p, reward, done)
    class FastOfflineDataset:
        def __init__(self, load_buff, capacity, device='cpu', length=None):

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

    file = open(config.load_buffer, 'rb')
    load_buff = pickle.load(file)
    file.close()

    tds = FastOfflineDataset(load_buff, length=config.buffer_steps, capacity=config.buffer_capacity)

    train_env = wandb_utils.LogRewards(train_env)

    if not config.silent:
        train_env = Plot(train_env, episodes_per_point=5, title=f'Train awac-{config.env_name}')

    """ test env """
    test_env = make_env()
    if not config.silent:
        test_env = Plot(test_env, episodes_per_point=1, title=f'Test awac-{config.env_name}')
    evaluator = helper.Evaluator(test_env)

    """ network """
    class AWACnet(nn.Module):
        """

        """
        def __init__(self, input_dims, actions, hidden_dims):
            super().__init__()
            self.q = nn.Sequential(nn.Linear(input_dims, hidden_dims), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dims, actions))
            self.policy = nn.Sequential(nn.Linear(input_dims, hidden_dims), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dims, actions))

        def forward(self, state):
            values = self.q(state)
            actions = self.policy(state)
            action_dist = Categorical(logits=log_softmax(actions, dim=1))
            return values, action_dist

    awac_net = AWACnet(
        input_dims=test_env.observation_space.shape[0],
        actions=test_env.action_space.n,
        hidden_dims=config.hidden_dim).to(config.device)

    q_optim = torch.optim.Adam(awac_net.q.parameters(), lr=config.optim_lr)
    policy_optim = torch.optim.Adam(awac_net.policy.parameters(), lr=config.optim_lr)

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        checkpoint.load(config.load, prefix='best', awac_net=awac_net, q_optim=q_optim)

    """ policy to run on environment """
    def policy(state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(config.device)
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
    wandb.run.summary['offline_steps'] = offline_steps
    on_policy = False
    print(f'OFF POLICY FOR {len(tds)} steps')
    timing = []

    for total_steps, (s, a, s_p, r, d, _) in enumerate(driver.step_environment(train_env, policy)):
        steps += 1
        if total_steps > config.max_steps:
            break

        if total_steps > offline_steps:
            tds.append((s, a, s_p, r, d))
            if not on_policy:
                print('on policy NOW')
                on_policy = True

        """ train offline after batch steps saved"""
        if len(tds) < config.batch_size:
            continue
        else:
            awac.train_fast(tds, awac_net, q_optim, policy_optim, batch_size=config.batch_size, device=config.device)
            steps = 0



        """ test  """
        if total_steps > config.test_steps * tests_run:
            tests_run += 1
            evaluator.evaluate(policy, config.run_dir, sample_n=config.test_samples, capture=config.test_capture,
                               params={'awac_net': awac_net, 'q_optim': q_optim, 'policy_optim': policy_optim})