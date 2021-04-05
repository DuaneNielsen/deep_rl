import torch
import torch.nn as nn
import gym
import env
import legacy.buffer as bf
import driver
from algos import reinforce
from distributions import ScaledTanhTransformedGaussian
import wandb
import wandb_utils
from config import ArgumentParser, exists_and_not_none
import checkpoint
import baselines.helper as helper
from gymviz import Plot


if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='reinforce')
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-d', '--device', type=str)
    parser.add_argument('-r', '--run_id', type=int, default=-1)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--silent', action='store_true', default=False)

    """ reproducibility """
    parser.add_argument('--seed', type=int, default=None)

    """ training loop control """
    parser.add_argument('--max_steps', type=int, default=2000000)
    parser.add_argument('--test_steps', type=int, default=100000)
    parser.add_argument('--test_episodes', type=int, default=10)

    """ resume settings """
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)

    """ environment """
    parser.add_argument('--env_name', type=str, default='CartPoleContinuous-v1')
    parser.add_argument('--env_render', action='store_true', default=False)

    """ hyper-parameters """
    parser.add_argument('--optim_class', type=str)
    parser.add_argument('--optim_lr', type=float, default=1e-4)
    parser.add_argument('--scheduler-class', type=str)
    parser.add_argument('--episodes_per_batch', type=int, default=16)
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--min_variance', type=float, default=0.05)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--clip_max', type=float, default=-0.1)
    parser.add_argument('--clip_min', type=float, default=-2.0)

    config = parser.parse_args()

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)

    wandb.init(project=f"reinforce-{config.env_name}", config=config)

    """ environment """
    def make_env():
        env = gym.make(config.env_name)
        assert len(env.observation_space.shape) == 1, "Only 1D continuous observation space is supported"
        assert len(env.action_space.shape) == 1, "Only 1D continuous action space is supported"
        if config.seed is not None:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
        return env

    train_env, train_buffer = bf.wrap(make_env())
    train_buffer.enrich(bf.DiscountedReturns(discount=config.discount))
    train_env = wandb_utils.LogRewards(train_env)
    if not config.silent:
        train_env = Plot(train_env, episodes_per_point=config.episodes_per_batch, title=f'Train reinforce-{config.env_name}')

    test_env = make_env()
    if not config.silent:
        test_env = Plot(test_env, episodes_per_point=1, title=f'Test reinforce-{config.env_name}')
    evaluator = helper.Evaluator(test_env)

    """ network """
    class PolicyNet(nn.Module):
        """
        policy(state) returns distribution over actions
        uses ScaledTanhTransformedGaussian as per Hafner
        """

        def __init__(self, input_dim, hidden_dim, output_dim=1, min=-1.0, max=1.0):
            super().__init__()
            self.mu = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dim, output_dim))
            self.scale = nn.Linear(input_dim, output_dim, bias=False)
            self.min = min
            self.max = max

        def forward(self, state):
            mu = self.mu(state)
            scale = torch.sigmoid(self.scale(state)) + config.min_variance
            return ScaledTanhTransformedGaussian(mu, scale, min=self.min, max=self.max)

    policy_net = PolicyNet(train_env.observation_space.shape[0],
                           config.hidden_dim,
                           train_env.action_space.shape[0],
                           min=test_env.action_space.low[0],
                           max=test_env.action_space.high[0])

    optim = torch.optim.Adam(policy_net.parameters(), lr=config.optim_lr)

    """ policy to run on environment """
    def policy(state):
        state = torch.from_numpy(state).float()
        action = policy_net(state)
        a = action.rsample()
        return a.numpy()

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        checkpoint.load(config.load, prefix='best', policy_net=policy_net, optim=optim)

    """ demo if command switch is set """
    evaluator.demo(config.demo, policy)

    """ begin training loop """
    total_steps = 0
    tests_run = 0

    while total_steps < config.max_steps:
        for ep in range(config.episodes_per_batch):
            driver.episode(train_env, policy)
        total_steps += len(train_buffer)

        """ train """
        reinforce.train(train_buffer, policy_net, optim, clip_min=config.clip_min, clip_max=config.clip_max)

        """ test """
        if total_steps > config.test_steps * tests_run:
            evaluator.evaluate(policy, config.run_dir, {'policy_net': policy_net, 'optim': optim}, config.test_episodes)
            tests_run += 1

