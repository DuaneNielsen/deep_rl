import torch
import torch.nn as nn

import gym
import env
import env.wrappers as wrappers

from torch.nn.functional import max_pool2d, softmax
from torch.distributions import Categorical
import numpy as np
from argparse import ArgumentParser
import driver
import algos.ppo as ppo
import rl
from rich.progress import Progress
from rich import print
import buffer_h5 as b5

if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='configuration switches')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--record', action='store_true', default=False)
    parser.add_argument('--filename', type=str, default='buffer.h5')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--complevel', type=int, default=6)
    parser.add_argument('--device', type=str, default='cpu')
    config = parser.parse_args()

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)

    """ environment """


    def make_env():
        env = gym.make('BreakoutDeterministic-v4')
        env = wrappers.EpisodicLifeEnv(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=4000)
        if 'NOOP' in env.unwrapped.get_action_meanings():
            env = wrappers.NoopResetEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = wrappers.FireResetEnv(env)
        env = wrappers.ClipState2D(env, 0, 24, 210 - 24, 160)
        env = wrappers.WarpFrame(env)
        env = wrappers.ScaledFloatFrame(env)
        env = wrappers.Gradient(env)

        if config.seed is not None:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
        return env


    env = make_env()
    actions = env.action_space.n

    """ network """


    class A2CNet(nn.Module):
        """
        policy(state) returns distribution over actions
        uses ScaledTanhTransformedGaussian as per Hafner
        """

        def __init__(self, hidden_dims, actions):
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
            a_dist = Categorical(action)
            return value, a_dist


    a2c_net = A2CNet(
        hidden_dims=64,
        actions=actions)
    a2c_net = ppo.PPOWrapModel(a2c_net).to(config.device)
    a2c_net.load_state_dict(torch.load('breakout.sd'))

    """ policy to run on environment """


    def policy(state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(config.device)
        value, action = a2c_net(state)
        a = action.sample()
        assert torch.isnan(a) == False
        return a.cpu().numpy()


    if config.record:

        with Progress() as progress:
            task = progress.add_task('recording', total=config.steps)

            buffer = b5.Buffer()
            state_col = b5.Column('state', (84, 84, 2), np.uint8, compression='gzip', compression_opts=6)
            raw_col = b5.Column('raw', (210, 160, 3), np.uint8, compression='gzip', compression_opts=6)
            action_col = b5.Column('action', dtype=np.int64, chunk_size=100000)
            buffer.create(config.filename, state_col=state_col, raw_col=raw_col, action_col=action_col)

            for step, s, a, s_p, r, d, i, m in buffer.step(env, policy, timing=True, capture_raw=True):
                if len(m) > 2:
                    print(f'{m}')
                progress.advance(task)
                if step == config.steps:
                    break
            buffer.close()

    else:
        """ demo  """
        while True:
            driver.episode(env, policy, render=True)
