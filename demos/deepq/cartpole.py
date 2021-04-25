import torch
from torch import nn
import gym
from gymviz import Plot
from argparse import ArgumentParser
import checkpoint
import rl
from demos import logging

if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='configuration switches')
    parser.add_argument('--silent', action='store_true', default=True)

    """ reproducibility """
    parser.add_argument('--seed', type=int, default=None)

    """ vizualization """
    parser.add_argument('--plot_episodes_per_point', type=int, default=32)

    """ logging """
    parser.add_argument('--project', type=str)
    parser.add_argument('--log_episodes', type=int, default=0)
    parser.add_argument('--test_steps', type=int, default=1000)
    parser.add_argument('--video_episodes', type=int, default=3)
    parser.add_argument('--test_episodes', type=int, default=16)
    parser.add_argument('--env_name', type=str, default='CartPole-v1')

    config = parser.parse_args()

    """ Environment """
    def make_env():
        env = gym.make('CartPole-v1')
        """ check environment has continuous input, discrete output"""
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert len(env.observation_space.shape) == 1
        assert isinstance(env.action_space, gym.spaces.Discrete)
        return env

    env = make_env()

    test_env = make_env()
    if not config.silent:
        test_env = Plot(test_env, episodes_per_point=1, title=f'Test deepq-{config.env_name}')

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)
        test_env.seed(config.seed)
        test_env.action_space.seed(config.seed)

    class QNet(nn.Module):
        """
        Simple MLP, takes in state and outputs a value for each action
        """

        def __init__(self, state_size, hidden_size, action_size):
            super().__init__()
            self.q_net = nn.Sequential(nn.Linear(state_size, hidden_size), nn.SELU(inplace=True),
                                       nn.Linear(hidden_size, action_size))

        def forward(self, state):
            action = self.q_net(state)
            return action

    q_net = QNet(state_size=env.observation_space.shape[0], hidden_size=16, action_size=env.action_space.n)

    checkpoint.load('.', prefix='cartpole', q_net=q_net)

    def exploit_policy(state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = torch.argmax(q_net(state), dim=1)
        return action.item()


    """ demo  """
    rl.demo(config.log_episodes == 0, test_env, exploit_policy)

    logging.log(config, env, exploit_policy, config.project, config.env_name, config.log_episodes,
                config.test_steps, config.video_episodes, config.test_episodes)