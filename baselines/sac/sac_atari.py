import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

import gym
import env
import env.wrappers as wrappers
from gymviz import Plot

from algos import sac
from torch.distributions import Categorical
from config import exists_and_not_none, ArgumentParser, EvalAction
import wandb
import wandb_utils
import checkpoint
from gym.wrappers.transform_reward import TransformReward
import capture
import os
import warnings

warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")


def rescale_reward(reward):
    return reward * config.env_reward_scale - config.env_reward_bias


if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='configuration switches')
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-d', '--device', type=str)
    parser.add_argument('-r', '--run_id', type=int, default=-1)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--precision', type=str, action=EvalAction, default=torch.float32)
    parser.add_argument('--worker_threads', type=int, default=2)

    """ reproducibility """
    parser.add_argument('--seed', type=int, default=None)

    """ main loop control """
    parser.add_argument('--max_steps', type=int, default=150000)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--test_steps', type=int, default=5000)
    parser.add_argument('--test_samples', type=int, default=5)
    parser.add_argument('--test_capture', action='store_true', default=True)
    parser.add_argument('--capture_freq', type=int, default=5000)

    """ resume settings """
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)

    """ environment """
    parser.add_argument('--env_name', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--env_render', action='store_true', default=False)
    parser.add_argument('--env_reward_scale', type=float, default=1.0)
    parser.add_argument('--env_reward_bias', type=float, default=0.0)
    parser.add_argument('--env_timelimit', type=int, default=3000)

    """ hyper-parameters """
    parser.add_argument('--optim_lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--replay_len', type=int, default=1000000)
    parser.add_argument('--q_update_ratio', type=int, default=2)


    config = parser.parse_args()

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)

    if 'DEVICE' in os.environ:
        config.device = os.environ['DEVICE']

    if config.debug:
        config.worker_threads=0

    wandb.init(project=f"sac-{config.env_name}", config=config)

    """ environment """
    def make_env():
        env = gym.make(config.env_name)
        env = wrappers.TimeLimit(env.unwrapped, max_episode_steps=config.env_timelimit)
        env = capture.VideoCapture(env, config.run_dir, freq=config.capture_freq)
        env = wrappers.EpisodicLifeEnv(env)
        if 'NOOP' in env.unwrapped.get_action_meanings():
            env = wrappers.NoopResetEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = wrappers.FireResetEnv(env)
        env = wrappers.ClipState2D(env, 0, 24, 210 - 24, 160)
        env = wrappers.WarpFrame(env)
        env = wrappers.ScaledFloatFrame(env)
        env = wrappers.Gradient(env)
        env = wrappers.ClipRewardEnv(env)
        env = wrappers.PenalizeOneIfDone(env)
        env = TransformReward(env, rescale_reward)

        if config.seed is not None:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
        return env


    """ training env with replay buffer """
    train_env = make_env()
    if config.debug:
        train_env = Plot(train_env, episodes_per_point=5, title=f'Train sac-{config.env_name}')

    """ test env """
    test_env = make_env()
    if config.debug:
        test_env = Plot(test_env, episodes_per_point=1, title=f'Test sac-{config.env_name}')


    class AtariVision(nn.Module):
        def __init__(self, feature_size=512):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.conv2[0].weight.data.mul_(2 ** -0.5)
            self.conv3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.conv3[0].weight.data.mul_(3 ** -0.5)
            self.conv4 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.conv4[0].weight.data.mul_(4 ** -0.5)
            self.conv5 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.conv5[0].weight.data.mul_(5 ** -0.5)
            self.conv6 = nn.Sequential(
                nn.Conv2d(256, feature_size, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.conv6[0].weight.data.mul_(6 ** -0.5)

            self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(5)])

        def forward(self, state):
            l1 = self.conv1(state.permute(0, 3, 1, 2))
            l2 = self.conv2(l1) + self.bias[0]
            l3 = self.conv3(l2) + self.bias[1]
            l4 = self.conv4(l3) + self.bias[2]
            l5 = self.conv5(l4) + self.bias[3]
            l6 = self.conv6(l5) + self.bias[4]
            return l6.flatten(start_dim=1)


    class MLP(nn.Module):
        def __init__(self, feature_size, hidden_dims, actions):
            super().__init__()
            self.l1 = nn.Sequential(nn.Linear(feature_size, hidden_dims), nn.SELU(inplace=True))
            self.l2 = nn.Sequential(nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True))
            self.l3 = nn.Linear(hidden_dims, actions, bias=False)
            self.l1[0].weight.data.mul_(7 ** -0.5)
            self.l1[0].bias.data.mul_(7 ** -0.5)
            self.l2[0].weight.data.mul_(8 ** -0.5)
            self.l2[0].bias.data.mul_(8 ** -0.5)
            self.l3.weight.data.zero_()
            self.gain = nn.Parameter(torch.ones(1))
            self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(3)])

        def forward(self, features):
            hidden = self.l1(features) + self.bias[0]
            hidden = self.l2(hidden) + self.bias[1]
            return self.l3(hidden) * self.gain + self.bias[2]


    class Q(nn.Module):
        def __init__(self, feature_size, hidden_dims, actions):
            super().__init__()
            self.q = nn.Sequential(
                AtariVision(feature_size),
                MLP(feature_size=feature_size, hidden_dims=hidden_dims, actions=actions))

        def forward(self, state):
            return self.q(state)


    class Policy(nn.Module):
        def __init__(self, feature_size, hidden_dims, actions):
            super().__init__()
            self.actions = actions
            self.policy = nn.Sequential(
                AtariVision(feature_size),
                MLP(feature_size=feature_size, hidden_dims=hidden_dims, actions=actions))

        def forward(self, state):
            return torch.softmax(self.policy(state), dim=-1)


    class QNet(nn.Module):
        def __init__(self, hidden_dims, actions, ensemble=2):
            super().__init__()
            self.q = [Q(256, hidden_dims, actions) for _ in range(ensemble)]

        def to(self, device):
            self.q = [q.to(device) for q in self.q]
            return self

        def parameters(self, recurse=True):
            params = []
            for q in self.q:
                for param in q.parameters():
                    params.append(param)
            return params

        def forward(self, state):
            values = []
            for q in self.q:
                values += [q(state)]
            values = torch.stack(values, dim=-1)
            min_q, _ = torch.min(values, dim=-1)
            return min_q


    assert isinstance(test_env.action_space, gym.spaces.Discrete), "action spaces is not discrete"
    assert len(test_env.observation_space.shape) == 3, "only image observation spaces are supported"

    q_net = QNet(
        actions=test_env.action_space.n,
        hidden_dims=config.hidden_dim).to(config.device)

    target_q_net = QNet(
        actions=test_env.action_space.n,
        hidden_dims=config.hidden_dim).to(config.device)

    policy_net = Policy(
        feature_size=256,
        actions=test_env.action_space.n,
        hidden_dims=config.hidden_dim
    ).to(config.device)

    q_optim = torch.optim.Adam(q_net.parameters(), lr=config.optim_lr)
    policy_optim = torch.optim.Adam(policy_net.parameters(), lr=config.optim_lr)
    wandb.watch(policy_net)
    wandb.watch(q_net)

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        checkpoint.load(config.load, prefix='best', q=q_net, q_optim=q_optim, policy=policy_net,
                        policy_optim=policy_optim)

    """ policy to run on environment """


    def policy(state):
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).to(config.device)
            probs = policy_net(state)
            assert ~torch.isnan(probs).any()
            a_dist = Categorical(probs=probs)
            wandb.log({'entropy': a_dist.entropy().item()}, step=wandb_utils.global_step)
            return a_dist.sample().item()


    """ policy to run on test environment """


    def exploit_policy(state):
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).to(config.device)
            dist = policy_net(state)
            assert ~torch.isnan(dist).any()
            a = torch.argmax(dist)
            return a.item()


    """ demo  """
    wandb_utils.demo(config.demo, env, policy)

    """ train loop """
    evaluator = wandb_utils.Evaluator()
    ds = wandb_utils.StateBufferDataset(maxlen=config.replay_len, statebuffer=wandb_utils.ZCompressedBuffer())
    dl = None

    for step, (s, a, s_p, r, d, i) in enumerate(
            wandb_utils.step_environment(train_env, policy, ds, render=config.env_render)):

        ds.append((s, a, s_p, r, d))

        if dl is None:
            dl = DataLoader(ds, batch_size=config.batch_size, sampler=RandomSampler(ds, replacement=True),
                            num_workers=config.worker_threads)

        if len(ds) < config.batch_size * config.q_update_ratio or len(ds) < config.warmup:
            continue  # sample at least a couple full batches for the first update

        sac.train_discrete(dl, q_net, target_q_net, policy_net, q_optim, policy_optim,
                           discount=config.discount, polyak=config.polyak, q_update_ratio=config.q_update_ratio,
                           alpha=config.alpha, device=config.device, precision=config.precision)

        if evaluator.evaluate_now(config.test_steps):
            evaluator.evaluate(test_env, exploit_policy, run_dir=config.run_dir, capture=config.test_capture,
                               params={'q': q_net, 'q_optim': q_optim, 'policy': policy_net,
                                       'policy_optim': policy_optim})

        if step > config.max_steps:
            break
