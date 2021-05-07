import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

import gym
import env
import env.wrappers as wrappers
from gymviz import Plot

from algos import cql
from config import exists_and_not_none, ArgumentParser, EvalAction
import wandb
import wandb_utils
import rl
import torch_utils
from rich.progress import track
import logs
import d4rl

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
    parser.add_argument('--test_steps', type=int, default=2000)
    parser.add_argument('--test_episodes', type=int, default=8)
    parser.add_argument('--load_buffer', type=str)
    parser.add_argument('--summary_video_episodes', type=int, default=0)

    """ resume settings """
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)

    """ environment """
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--env_render', action='store_true', default=False)
    parser.add_argument('--env_reward_scale', type=float, default=1.0)
    parser.add_argument('--env_reward_bias', type=float, default=0.0)
    parser.add_argument('--d4rl', action='store_true', default=False)

    """ hyper-parameters """
    parser.add_argument('--optim_lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.005)
    parser.add_argument('--policy_alpha', type=float, default=0.1)
    parser.add_argument('--policy_alpha_decay', type=float, default=3e-4)
    parser.add_argument('--cql_alpha', type=float, default=3.0)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--q_update_ratio', type=int, default=1)
    parser.add_argument('--min_return', type=float, default=-999999)

    config = parser.parse_args()

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)

    project = f"cql-{config.env_name}"
    wandb.init(project=project, config=config)
    logs.init(run_dir=config.run_dir)

    def make_env():
        """ environment """
        env = gym.make(config.env_name)
        env = wrappers.RescaleReward(env, config.env_reward_scale, config.env_reward_bias)
        if config.seed is not None:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
        if config.debug:
            env = Plot(env, episodes_per_point=1, title=f'Test cql-{config.env_name}')
        return env

    class MLP(nn.Module):
        def __init__(self, input_dims, hidden_dims, out_dims):
            super().__init__()
            self.hidden = nn.Sequential(nn.Linear(input_dims, hidden_dims), nn.SELU(inplace=True),
                                        nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                                        nn.Linear(hidden_dims, out_dims))

        def forward(self, state):
            return self.hidden(state)


    class Policy(nn.Module):
        def __init__(self, input_dims, hidden_dims, actions):
            super().__init__()
            self.actions = actions
            self.mlp = MLP(input_dims, hidden_dims, actions)

        def forward(self, state):
            return torch.softmax(self.mlp(state), dim=-1)


    class QNet(nn.Module):
        def __init__(self, input_dims, hidden_dims, actions, ensemble=2):
            super().__init__()
            self.q = [MLP(input_dims, hidden_dims, actions) for _ in range(ensemble)]

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
            values =  torch.stack(values, dim=-1)
            min_q, _ = torch.min(values, dim=-1)
            return min_q

    """ test env """
    test_env = make_env()

    assert isinstance(test_env.action_space, gym.spaces.Discrete), "action spaces is not discrete"
    assert len(test_env.observation_space.shape) == 1, "only 1-D observation spaces are supported"

    q_net = QNet(
        input_dims=test_env.observation_space.shape[0],
        actions=test_env.action_space.n,
        hidden_dims=config.hidden_dim).to(config.device)

    target_q_net = QNet(
        input_dims=test_env.observation_space.shape[0],
        actions=test_env.action_space.n,
        hidden_dims=config.hidden_dim).to(config.device)

    policy_net = Policy(
        input_dims=test_env.observation_space.shape[0],
        actions=test_env.action_space.n,
        hidden_dims=config.hidden_dim
    ).to(config.device)

    q_optim = torch.optim.Adam(q_net.parameters(), lr=1e-4)
    policy_optim = torch.optim.Adam(policy_net.parameters(), lr=3e-5)

    all_params = {'q': q_net, 'q_optim': q_optim, 'policy': policy_net, 'policy_optim': policy_optim}

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        torch_utils.load_checkpoint(config.load, prefix='best', **all_params)


    def policy(state):
        """ policy to run on environment """
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            action = torch.exp(policy_net(state))
            a = torch.distributions.Categorical(probs=action).sample()
            assert ~torch.isnan(a).any()
            return a.item()


    def exploit_policy(state):
        """ policy to run on eval environment """
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            action = torch.exp(policy_net(state))
            a = torch.argmax(action)
            assert ~torch.isnan(a).any()
            return a.item()

    """ demo  """
    rl.demo(config.demo, test_env, policy)

    """ load dataset """
    if config.d4rl:
        data = d4rl.qlearning_dataset(test_env)
        buffer = TensorDataset(*[torch.from_numpy(numpy_array) for key, numpy_array in data.items()])
    else:
        buffer = rl.load(config.load_buffer)

    dl = DataLoader(buffer, batch_size=config.batch_size, sampler=RandomSampler(buffer, replacement=True))

    """ train loop """
    test_number = 1

    for step in track(range(config.max_steps), description='Training'):

        policy_alpha = (1.0 - config.policy_alpha_decay) ** step * (1.0 - config.policy_alpha) + config.policy_alpha

        cql.train_discrete(dl, q_net, target_q_net, policy_net, q_optim, policy_optim,
                           discount=config.discount, polyak=config.polyak, policy_alpha=policy_alpha,
                           cql_alpha=config.cql_alpha, q_update_ratio=config.q_update_ratio,
                           device=config.device, precision=config.precision)

        """ test """
        if step > config.test_steps * test_number:
            improved = rl.evaluate(test_env, exploit_policy, sample_n=config.test_episodes)

            if improved:
                torch_utils.save_checkpoint(config.run_dir, 'best', **all_params)
            test_number += 1

            logs.write()

    """ post summary of best policy for the run """
    torch_utils.load_checkpoint(config.run_dir, prefix='best', **all_params)
    stats = rl.evaluate(test_env, exploit_policy, sample_n=config.test_episodes,
                        vid_sample_n=config.summary_video_episodes)
    logs.write()