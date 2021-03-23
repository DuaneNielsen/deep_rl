import torch
from torch import nn
import buffer as bf
import gym
import random
import algos.dqn as dqn
import driver
from gymviz import Plot
import wandb
import wandb_utils
from config import ArgumentParser, exists_and_not_none
import checkpoint
import baselines.helper as helper
from gym.wrappers.transform_reward import TransformReward


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

    """ resume settings """
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)

    """ main loop control """
    parser.add_argument('--max_steps', type=int, default=200000)
    parser.add_argument('--test_steps', type=int, default=20000)
    parser.add_argument('--test_episodes', type=int, default=10)

    """ environment """
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--env_render', action='store_true', default=False)
    parser.add_argument('--env_reward_scale', type=float, default=1.0)
    parser.add_argument('--env_reward_bias', type=float, default=0.0)

    """ vizualization """
    parser.add_argument('--plot_episodes_per_point', type=int, default=32)

    """ hyper-parameters """
    parser.add_argument('--optim_class', type=str)
    parser.add_argument('--optim_lr', type=float, default=1e-4)
    parser.add_argument('--scheduler-class', type=str)
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)


    config = parser.parse_args()

    wandb.init(project=f"dqn-{config.env_name}", config=config)


    """ Environment """
    def make_env():
        env = gym.make(config.env_name)
        env = TransformReward(env, rescale_reward)
        """ check environment has continuous input, discrete output"""
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert len(env.observation_space.shape) == 1
        assert isinstance(env.action_space, gym.spaces.Discrete)
        return env

    env = make_env()

    """ replay buffer """
    train_env, buffer = bf.wrap(env)
    if not config.silent:
        train_env = Plot(train_env, title=f'Train {config.env_name}', episodes_per_point=config.plot_episodes_per_point)
    train_env = wandb_utils.LogRewards(train_env)

    test_env = make_env()
    if not config.silent:
        test_env = Plot(test_env, episodes_per_point=1, title=f'Test deepq-{config.env_name}')
    evaluator = helper.Evaluator(test_env)

    """ random seed """
    if exists_and_not_none(config, 'seed'):
        torch.manual_seed(config.seed)
        train_env.seed(config.seed)
        train_env.action_space.seed(config.seed)
        test_env.seed(config.seed)
        test_env.action_space.seed(config.seed)

    class QNet(nn.Module):
        """
        Simple MLP, takes in state and outputs a value for each action
        """

        def __init__(self, state_size, hidden_size, action_size):
            super().__init__()
            self.q_net1 = nn.Sequential(nn.Linear(state_size, hidden_size), nn.SELU(inplace=True),
                                       nn.Linear(hidden_size, hidden_size), nn.SELU(inplace=True),
                                       nn.Linear(hidden_size, action_size))

            self.q_net2 = nn.Sequential(nn.Linear(state_size, hidden_size), nn.SELU(inplace=True),
                                       nn.Linear(hidden_size, hidden_size), nn.SELU(inplace=True),
                                       nn.Linear(hidden_size, action_size))
        def forward(self, state):
            v1 = self.q_net1(state)
            v2 = self.q_net2(state)
            value = torch.min(torch.stack([v1, v2], dim=-1), dim=-1)
            return value

    q_net = QNet(state_size=env.observation_space.shape[0], hidden_size=config.hidden_size, action_size=env.action_space.n)
    optim = torch.optim.SGD(q_net.parameters(), lr=config.optim_lr)

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        checkpoint.load(config.load, prefix='best', q_net=q_net, optim=optim)

    wandb.watch(q_net)

    """ epsilon greedy policy to run on environment """
    def policy(state):
        if random.random() < config.epsilon:
            return random.randint(0, 1)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            action = torch.argmax(q_net(state), dim=1)
            return action.item()

    def exploit_policy(state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = torch.argmax(q_net(state), dim=1)
        return action.item()


    """ demo  """
    if config.demo:
        while True:
            driver.episode(test_env, exploit_policy, render=True)
            buffer.clear()

    """ main loop """
    best_mean_return = -999999
    tests_run = 0

    """ execute 1 transition on environment """
    for step_n, _ in enumerate(driver.step_environment(train_env, policy, render=config.env_render)):
        if step_n < config.batch_size:
            continue
        if step_n > config.max_steps:
            break

        """ sample a batch and update """
        dqn.train(buffer, q_net, optim, batch_size=config.batch_size, discount=config.discount)

        """ periodically test  """
        if step_n > config.test_steps * tests_run:
            evaluator.evaluate(policy, config.run_dir, {'q_net': q_net, 'optim': optim}, config.test_episodes,
                               capture=True)
            tests_run += 1