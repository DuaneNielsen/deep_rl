import torch
import torch.nn as nn

import gym

import driver
from algos.awac import RecencyBiasSampler
from gymviz import Plot

import buffer as bf
from algos import awac
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from config import exists_and_not_none, ArgumentParser
import wandb
import wandb_utils
import checkpoint
import baselines.helper as helper
from torch.nn.functional import log_softmax
from gym.wrappers.transform_reward import TransformReward
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
    parser.add_argument('--lam', type=float, default=0.3)
    parser.add_argument('--recency', type=float, default=1.0)


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

    load_buff = bf.load(config.load_buffer)

    tds = awac.FastOfflineDataset(load_buff, length=config.buffer_steps, capacity=config.buffer_capacity)

    wandb_env = wandb_utils.LogRewards(train_env)
    train_env = wandb_env

    if not config.silent:
        train_env = Plot(train_env, episodes_per_point=5, title=f'Train awac-{config.env_name}')

    """ test env """
    test_env = make_env()
    if not config.silent:
        test_env = Plot(test_env, episodes_per_point=1, title=f'Test awac-{config.env_name}')
    evaluator = helper.Evaluator(test_env)


    class Policy(nn.Module):
        def __init__(self, input_dims, hidden_dims, actions):
            super().__init__()
            self.policy = nn.Sequential(nn.Linear(input_dims, hidden_dims), nn.SELU(inplace=True),
                                        nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                                        nn.Linear(hidden_dims, actions))

        def forward(self, state):
            return Categorical(logits=log_softmax(self.policy(state), dim=1))


    """ network """


    class AWACnet(nn.Module):
        """

        """

        def __init__(self, input_dims, actions, hidden_dims):
            super().__init__()
            self.q = nn.Sequential(nn.Linear(input_dims, hidden_dims), nn.SELU(inplace=True),
                                   nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                                   nn.Linear(hidden_dims, actions))
            self.policy = Policy(input_dims, hidden_dims, actions)

        def forward(self, state):
            values = self.q(state)
            action_dist = self.policy(state)
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
    num_workers = 0 if config.debug else 2

    dl = DataLoader(tds, batch_sampler=RecencyBiasSampler(tds, batch_size=config.batch_size, recency=config.recency), num_workers=num_workers)

    # dl = DataLoader(tds, batch_size=config.batch_size, batch_sampler=RandomSampler(tds, replacement=False),
    #                 num_workers=num_workers)

    wandb.run.summary['offline_steps'] = offline_steps
    on_policy = False
    print(f'OFF POLICY FOR {len(tds)} steps')

    for global_step, (s, a, s_p, r, d, _) in enumerate(driver.step_environment(train_env, policy)):
        wandb_env.global_step = global_step
        if global_step > config.max_steps:
            print(f'Ending after {global_step} steps')
            quit()

        if global_step > offline_steps:
            tds.append((s, a, s_p, r, d))
            if not on_policy:
                print('on policy NOW')
                on_policy = True

        """ train offline after batch steps saved"""
        if len(tds) < config.batch_size:
            continue
        else:
            awac.train_discrete(dl, awac_net, q_optim, policy_optim, lam=config.lam,
                            device=config.device, debug=config.debug, measure_kl=True, global_step=global_step)
            steps = 0

        """ test  """
        if global_step > config.test_steps * (tests_run + 1):
            tests_run += 1
            evaluator.evaluate(policy, config.run_dir, sample_n=config.test_samples, capture=config.test_capture,
                               params={'awac_net': awac_net, 'q_optim': q_optim, 'policy_optim': policy_optim},
                               global_step=global_step)
