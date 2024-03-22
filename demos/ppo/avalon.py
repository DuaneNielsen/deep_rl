from avalonsim import Action
import random
import gymnasium as gym
from models.mlp import MLP, ActionHead
import torch
import checkpoint
from algos.ppo import PPOWrapModel
from avalonsim.wrappers import NoTurnaroundWrapper
from argparse import ArgumentParser
from pathlib import Path
from time import time


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    env = gym.make('Avalon-v1')
    # env = NoTurnaroundWrapper(env)
    hidden_dim = 128
    enemy_policy_net = MLP(
        in_features=env.observation_space.shape[0],
        hidden_dims=hidden_dim,
        head=ActionHead(
            hidden_dims=hidden_dim,
            actions=env.action_space.n)
    )
    enemy_policy_net = PPOWrapModel(enemy_policy_net).cuda()

    player_policy_net = MLP(
        in_features=env.observation_space.shape[0],
        hidden_dims=hidden_dim,
        head=ActionHead(
            hidden_dims=hidden_dim,
            actions=env.action_space.n)
    )
    player_policy_net = PPOWrapModel(player_policy_net).cuda()

    # run_dir = 'C:/Users/Cabron/PycharmProjects/deep_rl/baselines/ppo/runs/run_255'

    # get the most recent run directory
    root_dir = Path('C:/Users/Cabron/PycharmProjects/deep_rl/baselines/ppo/runs').glob("*")
    dirs = [d for d in root_dir if d.is_dir()]
    dirs.sort(key=lambda d: d.stat().st_ctime, reverse=True)
    run_dir = dirs[0]

    checkpoint.load(
        run_dir, prefix='top',
        player_policy_net=player_policy_net,
        enemy_policy_net=enemy_policy_net
    )

    checkpoint_paths = checkpoint.checkpoint_paths(run_dir, prefix='top')
    player_path = run_dir / Path(checkpoint_paths[0])
    enemy_path = run_dir / Path(checkpoint_paths[1])

    prev_player_timestamp = player_path.stat().st_mtime
    prev_enemy_timestamp = enemy_path.stat().st_mtime


    def player_policy(state):
        with torch.no_grad():
            state = torch.from_numpy(state).type(torch.float32).unsqueeze(0).to("cuda")
            return player_policy_net(state).probs.argmax(1).item()


    def enemy_policy(state):
        with torch.no_grad():
            state = torch.from_numpy(state).type(torch.float32).unsqueeze(0).to("cuda")
            return enemy_policy_net(state).probs.argmax(1).item()

    import pygame

    running = True

    env.render_fps = 32
    env.render_speed = 30
    state, info = env.reset()
    # print(state)

    rgb = env.render()
    random.seed(42)

    trajectory = []
    done, trunc = False, False
    actions = []

    while running:

        player_timestamp = player_path.stat().st_mtime
        enemy_timestamp = enemy_path.stat().st_mtime

        if player_timestamp > prev_player_timestamp or enemy_timestamp > prev_enemy_timestamp:
            print('LOADING NEW POLICY')
            prev_player_timestamp = player_timestamp
            prev_enemy_timestamp = enemy_timestamp
            checkpoint.load(
                run_dir, prefix='last',
                player_policy_net=player_policy_net,
                enemy_policy_net=enemy_policy_net
            )
            state, info = env.reset()
            rgb = env.render()
            done = False
            trajectory = []

        for event in pygame.event.get():
            pass
        #
        #     actions = []
        #
        #     if event.type == pygame.KEYDOWN:
        #         enemy_action = Action(enemy_policy(state))
        #         if event.key == pygame.K_a:
        #             actions = [Action.BACKWARD, enemy_action]
        #         elif event.key == pygame.K_d:
        #             actions = [Action.FORWARD, enemy_action]
        #         elif event.key == pygame.K_SPACE:
        #             actions = [Action.ATTACK, enemy_action]
        #         elif event.key == pygame.K_s:
        #             actions = [Action.NOOP, enemy_action]
        #         elif event.key == pygame.K_t:
        #             print([[s[0].name, s[1].name] for s in trajectory])
        #         else:
        #             break
        #
        #     if event.type == pygame.QUIT:
        #         running = False


        actions = [Action(player_policy(state)), Action(enemy_policy(state))]

        if len(actions) == 2:
            trajectory += [actions]
            state, reward, done, trunc, info = env.step(actions)

            rgb = env.render()

            print(len(trajectory))

        if done or trunc:
            # print([[s[0].name, s[1].name] for s in trajectory])
            state, info = env.reset()
            rgb = env.render()
            done, trunc = False, False
            trajectory = []

    pygame.quit()
