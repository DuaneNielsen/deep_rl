import torch

import algos.utils
import avalonsim
from models.mlp import MLP, ValueHead, ActionHead
from torch.utils.data import DataLoader

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import env
from copy import deepcopy
from gymviz import Plot

from algos import ppo
from config import exists_and_not_none, ArgumentParser, EvalAction
import wandb
import wandb_utils
import checkpoint
import baselines.helper as helper
import os
from avalonsim.wrappers import InvertRewardWrapper, NoTurnaroundWrapper
from tqdm import tqdm
from collections import namedtuple
from models.visual import RLPlot

FullTransition = namedtuple('FullTransition', ['s', 'a', 's_p', 'r', 'd', 't', 'i'])

if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='configuration switches')
    parser.add_argument('-c', '--config', type=str, default='avalon.yaml')
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-r', '--run_id', type=int, default=-1)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--precision', type=str, action=EvalAction, default=torch.float32)

    """ reproducibility """
    parser.add_argument('--seed', type=int, default=None)

    """ main loop control """
    parser.add_argument('--max_steps', type=int, default=1000000)
    parser.add_argument('--test_epoch', type=int, default=100)
    parser.add_argument('--test_episodes', type=int, default=10)

    """ resume settings """
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)

    """ environment """
    parser.add_argument('--env_name', type=str, default='Avalon-v1')
    parser.add_argument('--env_render', action='store_true', default=False)

    """ hyper-parameters """
    parser.add_argument('--optim_lr', type=float, default=1.3e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--exploration_noise', type=float, default=0.0565)

    config = parser.parse_args()

    """ check for device in environment variable """
    if 'DEVICE' in os.environ:
        config.device = os.environ['DEVICE']

    """ random seed """
    if config.seed is not None:
        algos.utils.seed_all(config.seed)

    wandb.init(project=f"ppo-a2c-{config.env_name}", config=config)

    """ environment """


    def make_env():
        env = gym.make(config.env_name, render_mode='rgb_array')
        env = NoTurnaroundWrapper(env)
        if config.seed is not None:
            env.unwrapped.seed(config.seed)
            env.unwrapped.action_space.seed(config.seed)
        return env


    # def make_env():
    #     env = gym.make('IteratedRockPaperScissors-v1')
    #     return env

    """ training env with replay buffer """
    train_env = make_env()
    train_env = RecordEpisodeStatistics(train_env)
    if config.debug:
        train_env = Plot(train_env, episodes_per_point=20, title=f'Train ppo-a2c-{config.env_name}-{config.run_id}',
                         refresh_cooldown=5.0)

    """ test env """
    test_player_env = make_env()
    test_enemy_env = InvertRewardWrapper(make_env())
    test_player_env = RecordVideo(test_player_env, video_folder=config.run_dir + '/player')
    test_enemy_env = RecordVideo(test_enemy_env, video_folder=config.run_dir + '/enemy')

    if config.debug:
        test_env = Plot(test_player_env, episodes_per_point=config.test_episodes,
                        title=f'Test ppo-a2c-{config.env_name}-{config.run_id}',
                        refresh_cooldown=5.0)


    def make_net():
        """ network """
        value_net = MLP(
            in_features=train_env.observation_space.shape[0],
            hidden_dims=config.hidden_dim,
            head=ValueHead(
                hidden_dims=config.hidden_dim
            )
        ).to(config.device)

        value_optim = torch.optim.Adam(value_net.parameters(), lr=config.optim_lr, weight_decay=0.0)

        policy_net = MLP(
            in_features=train_env.observation_space.shape[0],
            hidden_dims=config.hidden_dim,
            head=ActionHead(
                hidden_dims=config.hidden_dim,
                actions=train_env.action_space.n)
        )

        policy_optim = torch.optim.Adam(policy_net.parameters(), lr=config.optim_lr, weight_decay=0.0)

        policy_net = ppo.PPOWrapModel(policy_net).to(config.device)
        return value_net, value_optim, policy_net, policy_optim


    player_value_net, player_value_optim, player_policy_net, player_policy_optim = make_net()
    enemy_value_net, enemy_value_optim, enemy_policy_net, enemy_policy_optim = make_net()

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        checkpoint.load(config.load, prefix='best',
                        player_value_net=player_value_net, player_value_optim=player_value_optim,
                        player_policy_net=player_policy_net, player_policy_optim=player_policy_optim,
                        enemy_value_net=player_value_net, enemy_value_optim=player_value_optim,
                        enemy_policy_net=player_policy_net, enemy_policy_optim=player_policy_optim,
                        )

    player_prev_policy_net, enemy_prev_policy_net = deepcopy(player_policy_net), deepcopy(enemy_policy_net)


    def player_policy(state, exploration_noise=0.):
        """ rollout against a frozen adversarial policy """

        with torch.no_grad():
            state = torch.from_numpy(state).type(config.precision).unsqueeze(0).to(config.device)
            if exploration_noise == 0.:
                return [
                    player_policy_net(state).probs.argmax(1).item(),
                    enemy_prev_policy_net(state).probs.argmax(1).item()
                ]
            else:
                return [
                    player_policy_net(state, exploration_noise).sample().item(),
                    enemy_prev_policy_net(state).probs.argmax(1).item()
                ]


    def enemy_policy(state, exploration_noise=0.):
        with torch.no_grad():
            state = torch.from_numpy(state).type(config.precision).unsqueeze(0).to(config.device)
            if exploration_noise == 0.:
                return [
                    player_prev_policy_net(state).probs.argmax(1).item(),
                    enemy_policy_net(state).probs.argmax(1).item()
                ]
            else:
                return [
                    player_prev_policy_net(state).probs.argmax(1).item(),
                    enemy_policy_net(state, exploration_noise).sample().item()
                ]


    player_explore_policy = lambda state: player_policy(state, config.exploration_noise)
    enemy_explore_policy = lambda state: enemy_policy(state, config.exploration_noise)

    """ demo  """
    helper.demo(config.demo, env, player_policy)

    """ training loop """
    buffer = []
    dl = DataLoader(buffer, batch_size=config.batch_size)
    player_evaluator = helper.Evaluator()
    enemy_evaluator = helper.Evaluator()


    def step_env(env, policy, state=None, done=True, render=False):
        if state is None or done:
            state, info = env.reset()
        action = policy(state)
        state_p, reward, done, truncated, info = env.step(action)
        if render:
            env.render()
        return FullTransition(state, action, state_p, reward, done, truncated, info)


    (s_p, info), d = train_env.reset(), False

    player_reward_ma = 0.
    enemy_reward_ma = 0.

    for epoch in tqdm(range(100000)):
        player_prev_policy_net.load_state_dict(player_policy_net.state_dict())
        enemy_prev_policy_net.load_state_dict(enemy_policy_net.state_dict())

        for player_updates in range(8):
            for _ in range(config.batch_size):
                s, a, s_p, r, d, _, i = step_env(train_env, player_explore_policy, s_p, d)
                buffer.append((s, a[0], s_p, r, d))
                player_reward_ma = player_reward_ma * 0.99 + r * 0.01
                if "episode" in i:
                    epi_stats = i["episode"]
                    wandb.log({key: epi_stats[old_key] for old_key, key in
                               [('r', 'player_returns'), ('l', 'player_epi_len'), ('t', 'player_epi_t')]})

            player_stats = ppo.train_a2c_stable(dl, player_value_net, player_value_optim, player_policy_net,
                                                player_policy_optim,
                                                discount=config.discount, device=config.device,
                                                precision=config.precision)
            buffer.clear()
            wandb.log(player_stats)

        for enemy_updates in range(8):
            for _ in range(config.batch_size):
                s, a, s_p, r, d, _, i = step_env(train_env, enemy_explore_policy, s_p, d)
                buffer.append((s, a[1], s_p, -r, d))
                enemy_reward_ma = enemy_reward_ma * 0.99 - r * 0.01

            ppo.train_a2c_stable(dl, enemy_value_net, enemy_value_optim, enemy_policy_net, enemy_policy_optim,
                                 discount=config.discount, device=config.device, precision=config.precision)
            buffer.clear()

        wandb.log({'player_reward_ma': player_reward_ma, 'enemy_reward_ma': enemy_reward_ma})


        def collect_episode(env, policy):
            env = RecordEpisodeStatistics(env)
            (s_p, i), d, t = env.reset(), False, False
            while not (d or t):
                s, a, s_p, r, d, t, i = step_env(env, policy, s_p, d)
            epi_stats = i['episode']
            return epi_stats['r'][0], epi_stats['l'][0], []


        """ test  """
        if player_evaluator.evaluate_now(epoch, config.test_epoch):
            player_evaluator._evaluate(
                lambda: collect_episode(test_player_env, player_policy),
                config.run_dir,
                params={
                    'player_policy_net': player_policy_net,
                    'player_policy_optim': player_policy_optim,
                    'player_value_net': player_value_net,
                    'player_value_optim': player_value_optim
                },
                prefix='player_', sample_n=config.test_episodes, capture=False)

        if enemy_evaluator.evaluate_now(epoch, config.test_epoch):
            enemy_evaluator._evaluate(
                lambda: collect_episode(test_enemy_env, enemy_policy), config.run_dir,
                params={
                    'enemy_policy_net': enemy_policy_net,
                    'enemy_policy_optim': enemy_policy_optim,
                    'enemy_value_net': enemy_value_net,
                    'enemy_value_optim': enemy_value_optim
                },
                prefix='enemy_', sample_n=config.test_episodes, capture=False)
