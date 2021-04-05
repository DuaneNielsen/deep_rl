import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import gym
import env

import driver
from gymviz import Plot

import buffer as bf
from algos import ppo
from distributions import ScaledTanhTransformedGaussian
from config import exists_and_not_none, ArgumentParser, EvalAction
import wandb
import wandb_utils
import checkpoint
import baselines.helper as helper

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
    parser.add_argument('--test_steps', type=int, default=30000)
    parser.add_argument('--test_episodes', type=int, default=10)

    """ resume settings """
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)

    """ environment """
    parser.add_argument('--env_name', type=str, default='CartPoleContinuous-v1')
    parser.add_argument('--env_render', action='store_true', default=False)

    """ hyper-parameters """
    parser.add_argument('--optim_lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=16)

    config = parser.parse_args()

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)

    wandb.init(project=f"ppp-a2c-{config.env_name}", config=config)

    """ environment """
    def make_env():
        env = gym.make(config.env_name)
        if config.seed is not None:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
        return env

    """ training env with replay buffer """
    train_env = make_env()
    train_env = wandb_utils.LogRewards(train_env)
    if config.debug:
        train_env = Plot(train_env, episodes_per_point=5, title=f'Train ppo-a2c-{config.env_name}')

    """ test env """
    test_env = make_env()
    if config.debug:
        test_env = Plot(test_env, episodes_per_point=1, title=f'Test ppo-a2c-{config.env_name}')

    """ network """
    class A2CNet(nn.Module):
        """
        policy(state) returns distribution over actions
        uses ScaledTanhTransformedGaussian as per Hafner
        """
        def __init__(self, input_dims, hidden_dims, min, max):
            super().__init__()
            self.min = min
            self.max = max
            self.mu = nn.Sequential(nn.Linear(input_dims, hidden_dims), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dims, 2))
            self.scale = nn.Linear(input_dims, 1, bias=False)

        def forward(self, state):
            output = self.mu(state)
            value = output[..., 0:1]
            mu = output[..., 1:2]
            scale = torch.sigmoid(self.scale(state)) + 0.01
            a_dist = ScaledTanhTransformedGaussian(mu, scale, min=self.min, max=self.max)
            return value, a_dist

    a2c_net = A2CNet(
        input_dims=test_env.observation_space.shape[0],
        hidden_dims=config.hidden_dim,
        min=test_env.action_space.low[0],
        max=test_env.action_space.high[0])
    optim = torch.optim.Adam(a2c_net.parameters(), lr=config.optim_lr)

    a2c_net = ppo.PPOWrapModel(a2c_net)

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        checkpoint.load(config.load, prefix='best', a2c_net=a2c_net, optim=optim)

    """ policy to run on environment """
    def policy(state):
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            value, action = a2c_net(state)
            a = action.rsample()
            assert torch.isnan(a) == False
            return a.numpy()

    """ demo  """
    helper.demo(config.demo, env, policy)

    """ train loop """
    evaluator = helper.Evaluator()
    buffer = []
    dl = DataLoader(buffer, batch_size=config.batch_size)

    for global_step, (s, a, s_p, r, d, i) in enumerate(bf.step_environment(train_env, policy)):

        buffer.append((s, a, s_p, r, d))

        """ train online after batch steps saved"""
        if len(buffer) == config.batch_size:
            ppo.train_a2c(dl, a2c_net, optim, discount=config.discount, batch_size=config.batch_size,
                          precision=config.precision)
            buffer.clear()

        """ test """
        if evaluator.evaluate_now(global_step, config.test_steps):
            evaluator.evaluate(test_env, policy, config.run_dir, global_step=global_step,
                               params={'a2c_net': a2c_net, 'optim': optim})

        if global_step > config.max_steps:
            break
