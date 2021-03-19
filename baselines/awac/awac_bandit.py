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
import numpy as np
from torch.nn.functional import log_softmax

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
    parser.add_argument('--env_name', type=str, default='Bandit-v1')
    parser.add_argument('--env_render', action='store_true', default=False)

    """ hyper-parameters """
    parser.add_argument('--optim_lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=16)

    config = parser.parse_args()

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)

    wandb.init(project=f"awac-{config.env_name}", config=config)

    """ environment """
    def make_env():
        env = gym.make(config.env_name)
        if config.seed is not None:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
        return env

    """ training env with replay buffer """
    train_env, train_buffer = bf.wrap(make_env())
    train_buffer.enrich(bf.DiscountedReturns(discount=config.discount))
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
            self.q = nn.Parameter(torch.randn(input_dims, actions))
            self.policy = nn.Parameter(torch.randn(input_dims, actions))
            # self.q = nn.Sequential(nn.Linear(input_dims, hidden_dims), nn.SELU(inplace=True),
            #                         nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
            #                         nn.Linear(hidden_dims, 2))
            # self.scale = nn.Linear(input_dims, 1, bias=False)

        def forward(self, state):
            i = torch.argmax(state, dim=1)
            values = self.q[i]
            actions = self.policy[i]
            action_dist = Categorical(logits=log_softmax(actions, dim=1))
            return values, action_dist

    awac_net = AWACnet(
        input_dims=test_env.observation_space.n,
        actions=test_env.action_space.n,
        hidden_dims=config.hidden_dim)
    q_optim = torch.optim.Adam([awac_net.q], lr=config.optim_lr)
    policy_optim = torch.optim.Adam([awac_net.policy], lr=config.optim_lr)

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        checkpoint.load(config.load, prefix='best', awac_net=awac_net, optim=q_optim)

    """ policy to run on environment """
    def policy(state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        value, action = awac_net(state)
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

        """ train offline after batch steps saved"""
        if steps < config.batch_size:
            continue
        else:
            awac.train_discrete(train_buffer, awac_net, critic_optim=q_optim, actor_optim=policy_optim, batch_size=config.batch_size)
            steps = 0
            print("QTABLE")
            print(awac_net.q)
            print("POLICY TABLE")
            print(Categorical(logits=log_softmax(awac_net.policy, dim=1)).probs)
        """ test  """
        if total_steps > config.test_steps * tests_run:
            tests_run += 1
            evaluator.evaluate(policy, config.run_dir, {'awac_net': awac_net, 'q_optim': q_optim, 'policy_optim': policy_optim})