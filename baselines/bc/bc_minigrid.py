import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from rich.progress import Progress
from rich import print

import gym
import d4rl
import pybulletgym
import env.wrappers as wrappers
from gymviz import Plot
import numpy as np

from algos import bc
from distributions import ScaledTanhTransformedGaussian
from config import exists_and_not_none, ArgumentParser, EvalAction, str2bool
import wandb
import wandb_utils
import checkpoint
import rl
import torch_utils
# from torchlars import LARS
import os
import logs

from mujoco_py import GlfwContext
GlfwContext(offscreen=True)


if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='configuration switches')
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-d', '--device', type=str)
    parser.add_argument('-r', '--run_id', type=int, default=-1)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--precision', type=str, action=EvalAction, default=torch.float32)
    parser.add_argument('--project', type=str)
    parser.add_argument('--tags', type=str, nargs='+', default=[])

    """ reproducibility """
    parser.add_argument('--seed', type=int, default=None)

    """ main loop control """
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--test_steps', type=int, default=10000)
    parser.add_argument('--test_episodes', type=int, default=16)
    parser.add_argument('--test_capture', action='store_true', default=False)
    parser.add_argument('--load_buffer', type=str)
    parser.add_argument('--summary_video_episodes', type=int, default=3)

    """ resume settings """
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)

    """ environment """
    parser.add_argument('--env_name', type=str, default='maze2d-umaze-v1')
    parser.add_argument('--env_render', action='store_true', default=False)
    parser.add_argument('--env_reward_scale', type=float, default=1.0)
    parser.add_argument('--env_reward_bias', type=float, default=0.0)

    """ hyper-parameters """
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--policy_lr', type=float, default=2e-5)
    parser.add_argument('--cql_alpha', type=float, default=3.0)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--max_variance', type=float, default=2.0)
    parser.add_argument('--min_variance', type=float, default=1e-5)

    config = parser.parse_args()

    if 'DEVICE' in os.environ:
        config.device = os.environ['DEVICE']

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)

    if config.debug:
        config.tags.append('debug')

    project = f"cql-v0.1-{config.env_name}" if config.project is None else config.project
    wandb.init(project=project, config=config, tags=config.tags)
    logs.init(run_dir=config.run_dir)
    print(f"checkpoints will be written to: [bold blue]{os.getcwd()}/{config.run_dir}[/bold blue]")


    def make_env():
        """ environment """
        env = gym.make(config.env_name)
        env = wrappers.RescaleReward(env, config.env_reward_scale, config.env_reward_bias)
        if config.seed is not None:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
        return env


    """ test env """
    test_env = make_env()
    if config.debug:
        test_env = Plot(test_env, episodes_per_point=1, title=f'Test cql-{config.env_name}')


    class SoftMLP(nn.Module):
        def __init__(self, input_dims, hidden_dims, out_dims):
            super().__init__()
            self.hidden = nn.Sequential(nn.Linear(input_dims, hidden_dims), nn.SELU(inplace=True),
                                        nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True))
            self.mu = nn.Linear(hidden_dims, out_dims)
            self.scale = nn.Linear(hidden_dims, out_dims, bias=False)
            self.mu.weight.data.zero_()
            self.mu.bias.data.zero_()
            self.scale.weight.data.uniform_(-0.001, 0.001)

        def forward(self, state):
            hidden = self.hidden(state)
            mu = self.mu(hidden)
            log_scale = self.scale(hidden)
            log_scale = torch.clamp(log_scale, config.min_variance, config.max_variance)
            scale = torch.exp(log_scale)
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

    assert np.all(test_env.action_space.low == test_env.action_space.low[0]), "action spaces do not have the same min"
    assert np.all(test_env.action_space.high == test_env.action_space.high[0]), "action spaces do not have the same max"
    assert len(test_env.observation_space.shape) == 1, "only 1-D observation spaces are supported"
    min_action, max_action = test_env.action_space.low[0].item(), test_env.action_space.high[0]

    policy_net = Policy(
        input_dims=test_env.observation_space.shape[0],
        actions=test_env.action_space.shape[0],
        hidden_dims=config.hidden_dim,
        min_action=min_action,
        max_action=max_action,
    ).to(config.device)

    policy_optim = torch.optim.Adam(policy_net.parameters(), lr=config.policy_lr)

    warmup = lambda epoch: min(1.0, epoch / config.warmup)
    policy_scheduler = torch.optim.lr_scheduler.LambdaLR(policy_optim, lr_lambda=warmup)

    networks_and_optimizers = {'policy': policy_net, 'policy_optim': policy_optim}

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        checkpoint.load(config.load, prefix='best', **networks_and_optimizers)


    def exploit_policy(state):
        """ policy to test """
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(config.device)
            action = policy_net(state)
            a = action.mean
            assert ~torch.isnan(a).any()
            return a.cpu().numpy()


    """ demo  """
    wandb_utils.demo(config.demo, test_env, exploit_policy)

    """ train loop """
    test_number = 1
    data = d4rl.qlearning_dataset(test_env)
    buffer = TensorDataset(*[torch.from_numpy(numpy_array) for key, numpy_array in data.items()])
    dl = DataLoader(buffer, batch_size=config.batch_size, sampler=torch_utils.RandomSampler(buffer, replacement=True))

    with Progress() as progress:

        t_train = progress.add_task('[red] steps ...', total=config.max_steps)
        t_eval = progress.add_task('[magenta] evals ...', total=config.max_steps//config.test_steps)

        for step in range(1, config.max_steps + 1):

            eval = step >= config.test_steps * test_number

            policy_alpha = (1.0 - config.policy_alpha_decay) ** step * (1.0 - config.policy_alpha_min) + config.policy_alpha_min

            bc.train_continuous(dl, policy_net, policy_optim,
                                 device=config.device, precision=config.precision, log=eval)
            policy_scheduler.step()

            """ test """
            if eval:
                improved = rl.evaluate(test_env, exploit_policy, sample_n=config.test_episodes)

                if improved:
                    torch_utils.save_checkpoint(config.run_dir, 'best', **networks_and_optimizers)
                test_number += 1

                logs.log({'warmup': warmup(step), 'trainer-Alpha': policy_alpha})
                logs.write()
                progress.update(t_eval, advance=1)

            progress.update(t_train, advance=1)

    """ post summary of best policy for the run """
    torch_utils.load_checkpoint(config.run_dir, prefix='best', **networks_and_optimizers)
    rl.evaluate(test_env, exploit_policy, sample_n=config.test_episodes, vid_sample_n=config.summary_video_episodes)
    logs.write()
