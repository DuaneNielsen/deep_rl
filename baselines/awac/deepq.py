import torch
from torch import nn
import gym
import driver
from argparse import ArgumentParser
import checkpoint
import buffer as bf
import pickle


if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='configuration switches')
    parser.add_argument('--silent', action='store_true', default=True)

    """ reproducibility """
    parser.add_argument('--seed', type=int, default=None)

    """ vizualization """
    parser.add_argument('--plot_episodes_per_point', type=int, default=32)

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
    env, buffer = bf.wrap(env)

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)
        env.seed(config.seed)
        env.action_space.seed(config.seed)

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

    for i in range(200):
        driver.episode(env, exploit_policy)

    file = open('./buffer.pkl', mode='wb')
    pickle.dump(buffer, file)
    file.close()