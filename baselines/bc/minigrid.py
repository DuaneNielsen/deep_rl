import gym
import matplotlib.pyplot as plt
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgObsWrapper
import matplotlib
matplotlib.use('TkAgg')
from buffer import ReplayBuffer, ReplayBufferDataset
from driver import step_environment
from algos import bc
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from env.wrappers import ApplyFunc
import numpy as np
import baselines.helper
from config import ArgumentParser
import wandb
import os
from rich import print


def prepro_obs(obs):
    """
    Converts the grid encoding from H, W, C to C, H, W and normalizes values in range 0-1

    Args:
        obs: observation

    Returns: normalized observation

    """
    obs = obs.transpose(2, 0, 1)
    return obs / 10.0


class PolicyNet(nn.Module):
    """
    policy(state) returns a score for each action
    """

    def __init__(self, linear_in_dims, actions):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(3, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 3, 1, 1)
        )
        self.output = nn.Linear(linear_in_dims, actions, bias=False)
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, state):
        z = self.vision(state).flatten(start_dim=1)
        return self.output(z)


if __name__ == '__main__':
    """ configuration """
    parser = ArgumentParser(description='configuration switches')
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-d', '--device', type=str)
    parser.add_argument('-r', '--run_id', type=int, default=-1)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--tags', type=str, nargs='+', default=[])

    """ reproducibility """
    parser.add_argument('--seed', type=int, default=None)

    """ main loop control """
    parser.add_argument('--max_steps', type=int, default=20)
    parser.add_argument('--test_episodes', type=int, default=2)

    """ resume settings """
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)

    """ environment """
    parser.add_argument('--env_name', type=str, default='MiniGrid-Empty-8x8-v0')
    parser.add_argument('--env_render', action='store_true', default=False)
    parser.add_argument('--env_reward_scale', type=float, default=1.0)
    parser.add_argument('--env_reward_bias', type=float, default=0.0)

    """ hyper-parameters """
    parser.add_argument('--optim_lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=16)

    config = parser.parse_args()


    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)

    wandb.init(project=f"bc-v0.1-{config.env_name}", config=config, tags=config.tags)
    print(f'checkpoints will be written to: [bold blue]{os.getcwd()}/{config.run_dir}[/bold blue]')

    """ environment """
    env = gym.make(config.env_name, new_step_api=True)
    env = FullyObsWrapper(env) # Use full observations
    env = ImgObsWrapper(env)
    env = ApplyFunc(env, prepro_obs, gym.spaces.Space(shape=(3, 8, 8), dtype=np.float32))
    env.unwrapped.max_steps = config.max_steps


    class OpenLoopPolicy:
        """ behaviour policy"""
        def __init__(self, action_seq):
            self.action_seq = action_seq
            self.i = 0

        def __call__(self, s):
            a = self.action_seq[self.i]
            self.i += 1
            return a


    behavior = [env.actions.forward] * 5 + [env.actions.right] + [env.actions.forward] * 5
    behavior_policy = OpenLoopPolicy(behavior)

    """ capture a dataset """
    buffer = ReplayBuffer()
    for s, a, s_p, r, d, info in step_environment(env, behavior_policy, render=True):
        buffer.append(s, a, s_p, r, d, info)
        if d:
            break

    ds = ReplayBufferDataset(buffer)
    dl = DataLoader(ds, batch_size=10)

    """ clone the behaviour """
    policy_net = PolicyNet(linear_in_dims=16, actions=3)
    optim = Adam(policy_net.parameters(), lr=config.optim_lr)


    def policy_net_eval(s):
        """ wrap the policy for inference"""
        with torch.no_grad():
            s = torch.from_numpy(s).to(policy_net.dummy_param.dtype)
            a = policy_net(s.unsqueeze(0))
            return torch.argmax(a).item()


    evaluator = baselines.helper.Evaluator()

    for epoch in range(100):
        bc.train_discrete(dl, policy_net, optim)
        mean_return, stdev_return = evaluator.evaluate(
            env, policy_net_eval, sample_n=config.test_episodes, render=True, run_dir=config.run_dir,
            params={'policy_net': policy_net, 'optim': optim}
        )
        print(f'[blue]{mean_return}[/blue]')
