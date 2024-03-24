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
from collections import namedtuple, deque
from models.visual import RLPlot
import numpy as np
from statistics import mean

FullTransition = namedtuple('FullTransition', ['s', 'a', 's_p', 'r', 'd', 't', 'i'])


class Dojo:
    def __init__(self, size):
        self.top_n_players = []
        self.top_n_enemy = []
        self.scores = np.full((size+1, size+1), fill_value=-np.inf)
        """
                   player        
              |    |    |    |
        enemy |    |    |    |
              |    |    |    |
              
        """
        self.size = size

    def evaluate(self, last_player, last_enemy, make_env):

        self.top_n_players += [last_player]
        self.top_n_enemy += [last_enemy]

        def policy(state, player_eval_policy, enemy_eval_policy):
            state = torch.from_numpy(state).type(config.precision).unsqueeze(0).to(config.device)
            return player_eval_policy(state).sample().item(), enemy_eval_policy(state).sample().item()

        """ evaluate the player against all the enemies"""
        for i, enemy_policy in enumerate(self.top_n_enemy):
            eval_env = make_env()

            def eval_policy(state):
                return policy(state, last_player, enemy_policy)

            all_returns, all_lengths = [], []
            for _ in range(5):
                returns, l, _ = collect_episode(eval_env, eval_policy)
                all_returns.append(returns)
                all_lengths.append(l)
            self.scores[i, len(self.top_n_players)-1] = mean(all_returns)

        """ evalute the enemy against all the players"""
        for i, player_policy in enumerate(self.top_n_players):
            eval_env = make_env()

            def eval_policy(state):
                return policy(state, player_policy, last_enemy)

            all_returns, all_lengths = [], []
            for _ in range(5):
                returns, l, _ = collect_episode(eval_env, eval_policy)
                all_returns.append(returns)
                all_lengths.append(l)
            self.scores[len(self.top_n_enemy)-1, i] = mean(all_returns)

        print()
        print(self.scores)
        print("player_marginals", self.scores.sum(0))
        print("enemy marginals", -self.scores.sum(1))

        player_rankings = np.argsort(-self.scores.sum(0))
        enemy_rankings = np.argsort(self.scores.sum(1))
        self.scores = self.scores[:, player_rankings]
        self.scores = self.scores[enemy_rankings, :]

        print(self.scores)
        print("ranked player", self.scores.sum(0)/self.scores.shape[1])
        print("ranked enemy", -self.scores.sum(1)/self.scores.shape[0])
        print("expected score", self.scores[0, 0])
        wandb.log({'dojo - expected score': self.scores[0, 0]})

        self.top_n_players = [self.top_n_players[i] for i in player_rankings[:len(self.top_n_enemy)]]
        self.top_n_enemy = [self.top_n_enemy[i] for i in enemy_rankings[:len(self.top_n_players)]]

        if len(self.top_n_players) > self.size:
            self.top_n_players.pop(-1)
            self.top_n_enemy.pop(-1)

        return self.top_n_players, self.top_n_enemy



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
        # env = NoTurnaroundWrapper(env)
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
    test_player_env = RecordVideo(test_player_env, video_folder=config.run_dir + '/player',
                                  episode_trigger=lambda episode_id: episode_id % 2 == 0)
    test_enemy_env = RecordVideo(test_enemy_env, video_folder=config.run_dir + '/enemy',
                                 episode_trigger=lambda episode_id: episode_id % 2 == 0)

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

    dojo = Dojo(10)

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        checkpoint.load(config.load, prefix='best',
                        player_value_net=player_value_net, player_value_optim=player_value_optim,
                        player_policy_net=player_policy_net, player_policy_optim=player_policy_optim,
                        enemy_value_net=player_value_net, enemy_value_optim=player_value_optim,
                        enemy_policy_net=player_policy_net, enemy_policy_optim=player_policy_optim,
                        )
    else:
        checkpoint.save(
            config.run_dir, prefix='last',
            player_value_net=player_value_net, player_value_optim=player_value_optim,
            player_policy_net=player_policy_net, player_policy_optim=player_policy_optim,
            enemy_value_net=player_value_net, enemy_value_optim=player_value_optim,
            enemy_policy_net=player_policy_net, enemy_policy_optim=player_policy_optim,
        )

    player_prev_policy_net, enemy_prev_policy_net = deepcopy(player_policy_net), deepcopy(enemy_policy_net)


    # def player_policy(state, exploration_noise=0.):
    #     """ rollout against a frozen adversarial policy """
    #
    #     with torch.no_grad():
    #         state = torch.from_numpy(state).type(config.precision).unsqueeze(0).to(config.device)
    #         if exploration_noise == 0.:
    #             return [
    #                 player_policy_net(state).probs.argmax(1).item(),
    #                 enemy_prev_policy_net(state).probs.argmax(1).item()
    #             ]
    #         else:
    #             return [
    #                 player_policy_net(state, exploration_noise).sample().item(),
    #                 enemy_prev_policy_net(state).probs.argmax(1).item()
    #             ]


    def player_policy(state, exploration_noise=0.):
        with torch.no_grad():
            state = torch.from_numpy(state).type(config.precision).unsqueeze(0).to(config.device)
            return [
                player_policy_net(state, exploration_noise).sample().item(),
                enemy_policy_net(state, 0).sample().item()
            ]


    # def enemy_policy(state, exploration_noise=0.):
    #     with torch.no_grad():
    #         state = torch.from_numpy(state).type(config.precision).unsqueeze(0).to(config.device)
    #         if exploration_noise == 0.:
    #             return [
    #                 player_prev_policy_net(state).probs.argmax(1).item(),
    #                 enemy_policy_net(state).probs.argmax(1).item()
    #             ]
    #         else:
    #             return [
    #                 player_prev_policy_net(state).probs.argmax(1).item(),
    #                 enemy_policy_net(state, exploration_noise).sample().item()
    #             ]


    def enemy_policy(state, exploration_noise=0.):
        with torch.no_grad():
            state = torch.from_numpy(state).type(config.precision).unsqueeze(0).to(config.device)
            return [
                player_policy_net(state, 0).sample().item(),
                enemy_policy_net(state, exploration_noise).sample().item()
            ]


    player_explore_policy = lambda state: player_policy(state, config.exploration_noise)
    enemy_explore_policy = lambda state: enemy_policy(state, config.exploration_noise)

    """ training loop """
    buffer = []
    dl = DataLoader(buffer, batch_size=config.batch_size)

    def step_env(env, policy, state=None, done=True, truncated=False, render=False):
        if state is None or done or truncated:
            state, info = env.reset()
        action = policy(state)
        state_p, reward, done, truncated, info = env.step(action)
        if render:
            env.render()
        return FullTransition(state, action, state_p, reward, done, truncated, info)


    def collect_episode(env, policy):
        env = RecordEpisodeStatistics(env)
        (s_p, i), d, t = env.reset(), False, False
        while not (d or t):
            s, a, s_p, r, d, t, i = step_env(env, policy, s_p, d)
        epi_stats = i['episode']
        return epi_stats['r'][0], epi_stats['l'][0], []

    def write_trajectory_stats(i, prefix):
        if "episode" in i:
            epi_stats = i["episode"]
            wandb.log({key: epi_stats[old_key] for old_key, key in
                       [('r', f'{prefix}_returns'), ('l', f'{prefix}_epi_len'), ('t', f'{prefix}_epi_t')]})

    (s_p, info), d, tr = train_env.reset(), False, False

    player_reward_ma = 0.
    enemy_reward_ma = 0.

    for epoch in tqdm(range(100000)):
        player_prev_policy_net.load_state_dict(player_policy_net.state_dict())
        enemy_prev_policy_net.load_state_dict(enemy_policy_net.state_dict())

        for player_updates in range(8):
            for _ in range(config.batch_size):
                s, a, s_p, r, d, tr, i = step_env(train_env, player_explore_policy, s_p, d, tr)
                buffer.append((s, a[0], s_p, r, d))
                write_trajectory_stats(i, 'player')

            player_stats = ppo.train_a2c_stable(dl, player_value_net, player_value_optim, player_policy_net,
                                                player_policy_optim,
                                                discount=config.discount, device=config.device,
                                                precision=config.precision)
            buffer.clear()
            wandb.log(player_stats)

        for enemy_updates in range(8):
            for _ in range(config.batch_size):
                s, a, s_p, r, d, tr, i = step_env(train_env, enemy_explore_policy, s_p, d, tr)
                buffer.append((s, a[1], s_p, -r, d))
                write_trajectory_stats(i, 'enemy')

            ppo.train_a2c_stable(dl, enemy_value_net, enemy_value_optim, enemy_policy_net, enemy_policy_optim,
                                 discount=config.discount, device=config.device, precision=config.precision)
            buffer.clear()

        if epoch % config.test_epoch == 0:

            top_n_players, top_n_enemy = dojo.evaluate(player_policy_net, enemy_policy_net, make_env)

            checkpoint.save(config.run_dir, prefix='top',
                            player_policy_net=top_n_players[0],
                            enemy_policy_net=top_n_enemy[0],
                            )