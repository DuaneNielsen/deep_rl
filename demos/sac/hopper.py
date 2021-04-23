import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import gym
import env.wrappers as wrappers
from gymviz import Plot
import numpy as np
import pybulletgym

from distributions import ScaledTanhTransformedGaussian
from config import exists_and_not_none, ArgumentParser, EvalAction
import wandb
import wandb_utils
import checkpoint
import rl
import torch_utils
from rich.progress import Progress

if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='configuration switches')
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-d', '--device', type=str)
    parser.add_argument('-r', '--run_id', type=int, default=-1)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--precision', type=str, action=EvalAction, default=torch.float32)

    """ reproducibility """
    parser.add_argument('--seed', type=int, default=None)

    """ main loop control """
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--test_steps', type=int, default=10000)
    parser.add_argument('--test_episodes', type=int, default=16)
    parser.add_argument('--test_capture', action='store_true', default=False)
    parser.add_argument('--log_episodes', type=int, default=0)
    parser.add_argument('--video_episodes', type=int, default=5)

    """ resume settings """
    parser.add_argument('--demo', action='store_true', default=True)
    #parser.add_argument('-l', '--load', type=str, default='runs/run_145')
    parser.add_argument('-l', '--load', type=str, default='runs/run_186')

    """ environment """
    parser.add_argument('--env_name', type=str, default='HopperPyBulletEnv-v0')
    parser.add_argument('--env_render', action='store_true', default=True)
    parser.add_argument('--env_reward_scale', type=float, default=1.0)
    parser.add_argument('--env_reward_bias', type=float, default=0.0)

    """ hyper-parameters """
    parser.add_argument('--optim_lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--min_variance', type=float, default=0.01)

    config = parser.parse_args()

    config.demo = config.log_episodes == 0

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)

    """ environment """
    def make_env():
        env = gym.make(config.env_name)
        env = wrappers.RescaleReward(env, config.env_reward_scale, config.env_reward_bias)
        if config.seed is not None:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
        return env

    """ training env with replay buffer """
    env = make_env()
    if config.debug:
        env = Plot(env, episodes_per_point=5, title=f'Demo-{config.env_name}')


    class SoftMLP(nn.Module):
        def __init__(self, input_dims, hidden_dims, out_dims):
            super().__init__()
            self.hidden = nn.Sequential(nn.Linear(input_dims, hidden_dims), nn.SELU(inplace=True),
                                        nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True))
            self.mu = nn.Linear(hidden_dims, out_dims)
            self.scale = nn.Linear(hidden_dims, out_dims)

        def forward(self, state):
            hidden = self.hidden(state)
            mu = self.mu(hidden)
            scale = torch.sigmoid(self.scale(hidden)) + config.min_variance
            return mu, scale


    class Policy(nn.Module):
        def __init__(self, input_dims, hidden_dims, actions, min_action, max_action):
            super().__init__()
            self.soft_mlp = SoftMLP(input_dims, hidden_dims, actions)
            self.min = min_action
            self.max = max_action

        def forward(self, state):
            mu, scale = self.soft_mlp(state)
            return ScaledTanhTransformedGaussian(mu, scale, min=self.min, max=self.max)


    class QNet(nn.Module):
        def __init__(self, input_dims, hidden_dims, actions, ensemble=2):
            super().__init__()
            self.q = [nn.Sequential(nn.Linear(input_dims + actions, hidden_dims), nn.SELU(inplace=True),
                                   nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                                   nn.Linear(hidden_dims, 1)) for _ in range(ensemble)]

        def parameters(self, recurse=True):
            params = []
            for q in self.q:
                for param in q.parameters():
                    params.append(param)
            return params

        def forward(self, state, action):
            sa = torch.cat((state, action), dim=1)
            values = []
            for q in self.q:
                values += [q(sa)]
            values = torch.stack(values, dim=-1)
            min_q, _ = torch.min(values, dim=-1)
            return min_q

    q_net = QNet(
        input_dims=env.observation_space.shape[0],
        actions=env.action_space.shape[0],
        hidden_dims=config.hidden_dim).to(config.device)

    target_q_net = QNet(
        input_dims=env.observation_space.shape[0],
        actions=env.action_space.shape[0],
        hidden_dims=config.hidden_dim).to(config.device)

    assert np.all(env.action_space.low == env.action_space.low[0]), "action spaces do not have the same min"
    assert np.all(env.action_space.high == env.action_space.high[0]), "action spaces do not have the same max"
    assert len(env.observation_space.shape) == 1, "only 1-D observation spaces are supported"

    policy_net = Policy(
        input_dims=env.observation_space.shape[0],
        actions=env.action_space.shape[0],
        hidden_dims=config.hidden_dim,
        min_action=env.action_space.low[0].item(),
        max_action=env.action_space.high[0].item()
    ).to(config.device)

    q_optim = torch.optim.Adam(q_net.parameters(), lr=config.optim_lr)
    policy_optim = torch.optim.Adam(policy_net.parameters(), lr=config.optim_lr)

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        checkpoint.load(config.load, prefix='best', q=q_net, q_optim=q_optim, policy=policy_net, policy_optim=policy_optim)

    """ policy to run on environment """
    def policy(state):
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            action = policy_net(state)
            a = action.rsample()
            assert ~torch.isnan(a).any()
            return a.numpy()

    """ policy to run on environment """
    def exploit_policy(state):
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            action = policy_net(state)
            a = action.mean
            assert ~torch.isnan(a).any()
            return a.numpy()


    """ demo  """
    wandb_utils.demo(config.demo, env, policy)

    """ logging loop """
    wandb.init(project=f"cql-{config.env_name}", config=config)
    buffer = rl.ReplayBuffer()
    dl = DataLoader(buffer, batch_size=config.batch_size, sampler=torch_utils.RandomSampler(buffer, replacement=True))

    episodes_captured = 0
    vidstream = []
    test_number = 1

    with Progress() as progress:
        run = progress.add_task('Generating', total=config.log_episodes)
        for step, s, a, s_p, r, d, i, m in rl.step(env, policy, buffer, render=True):

            if episodes_captured < config.log_episodes:
                buffer.append(s, a, s_p, r, d)
                if episodes_captured < config.video_episodes:
                    vidstream.append(m['frame'])
                else:
                    rl.global_render = False
                episodes_captured += 1 if d else 0
                progress.update(run, advance=1 if d else 0)
            else:
                break

            """ test """
            if step > config.test_steps * test_number:
                stats = rl.evaluate(env, policy, sample_n=config.test_episodes)
                wandb_utils.log_test_stats(stats)
                test_number += 1

    """ log transitions """
    filename = f'./{config.env_name}_{len(buffer)}.pkl'
    rl.save(buffer, filename)
    wandb.run.tags = [*wandb.run.tags, filename]

    """ log video """
    video_filename = f'./{config.env_name}_{len(buffer)}.mp4'
    torch_utils.write_mp4(video_filename, vidstream)
    wandb.log({'video': wandb.Video(video_filename, fps=4, format="mp4")})
