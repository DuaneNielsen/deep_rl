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
    parser.add_argument('--max_steps', type=int, default=200000)
    parser.add_argument('--test_steps', type=int, default=30000)
    parser.add_argument('--test_episodes', type=int, default=10)

    """ resume settings """
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)

    """ environment """
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--env_render', action='store_true', default=False)

    """ hyper-parameters """
    parser.add_argument('--optim_lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
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

    file = open('buffer.pkl', 'rb')
    load_buff = pickle.load(file)
    file.close()

    train_buffer.buffer = load_buff.buffer
    train_buffer.transitions = load_buff.transitions
    train_buffer.trajectories = load_buff.trajectories
    train_buffer.trajectory_info = load_buff.trajectory_info
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
        hidden_dims=config.hidden_dim)

    q_optim = torch.optim.Adam(awac_net.q.parameters(), lr=config.optim_lr)
    policy_optim = torch.optim.Adam(awac_net.policy.parameters(), lr=config.optim_lr)

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        checkpoint.load(config.load, prefix='best', awac_net=awac_net, q_optim=q_optim)

    """ policy to run on environment """
    def policy(state):
        state = torch.from_numpy(state).float().unsqueeze(0)
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

    offline_steps = len(train_buffer)
    wandb.run.summary['offline_steps'] = offline_steps
    train_buffer.record = False
    print(f'OFF POLICY FOR {len(train_buffer)} steps')

    for total_steps, _ in enumerate(driver.step_environment(train_env, policy)):
        steps += 1
        if total_steps > config.max_steps:
            break

        if total_steps > offline_steps:
            if not train_buffer.record:
                print('Recording NOW')
                train_buffer.clear()
            train_buffer.record = True

        """ train offline after batch steps saved"""
        if len(train_buffer) < config.batch_size:
            continue
        else:
            awac.train_discrete(train_buffer, awac_net, q_optim, policy_optim, batch_size=config.batch_size)
            steps = 0

        """ test  """
        if total_steps > config.test_steps * tests_run:
            tests_run += 1
            evaluator.evaluate(policy, config.run_dir,
                               {'awac_net': awac_net, 'q_optim': q_optim, 'policy_optim': policy_optim})