from avalonsim import Action
import random
import gymnasium as gym
from models.mlp import MLP, ActionHead
import torch
import checkpoint
from algos.ppo import PPOWrapModel
from avalonsim.wrappers import NoTurnaroundWrapper
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    env = gym.make('Avalon-v1')
    env = NoTurnaroundWrapper(env)
    hidden_dim = 128
    enemy_policy_net = MLP(
        in_features=env.observation_space.shape[0],
        hidden_dims=hidden_dim,
        head=ActionHead(
            hidden_dims=hidden_dim,
            actions=env.action_space.n)
    )
    enemy_policy_net = PPOWrapModel(enemy_policy_net)

    player_policy_net = MLP(
        in_features=env.observation_space.shape[0],
        hidden_dims=hidden_dim,
        head=ActionHead(
            hidden_dims=hidden_dim,
            actions=env.action_space.n)
    )
    player_policy_net = PPOWrapModel(player_policy_net)

    run_dir = 'C:/Users/Cabron/PycharmProjects/deep_rl/baselines/ppo/runs/run_253'

    checkpoint.load(
        run_dir, prefix='best',
        player_policy_net=player_policy_net,
        enemy_policy_net=enemy_policy_net
    )

    checkpoint_paths = checkpoint.checkpoint_paths(run_dir, prefix='best')

    prev_player_timestamp = checkpoint_paths['player_policy_net'].stat().st_mtime
    prev_enemy_timestamp = checkpoint_paths['enemy_policy_net'].stat().st_mtime


    def player_policy(state):
        with torch.no_grad():
            state = torch.from_numpy(state).type(torch.float32).unsqueeze(0)
            return player_policy_net(state).probs.argmax(1).item()


    def enemy_policy(state):
        with torch.no_grad():
            state = torch.from_numpy(state).type(torch.float32).unsqueeze(0)
            return enemy_policy_net(state).probs.argmax(1).item()

    import pygame

    running = True

    state, info = env.reset()
    print(state)

    rgb = env.render()
    random.seed(42)

    trajectory = []
    done = False
    actions = []

    while running:

        player_timestamp = checkpoint_paths['player_policy_net'].stat().st_mtime
        enemy_timestamp = checkpoint_paths['enemy_policy_net'].stat().st_mtime

        if player_timestamp > prev_player_timestamp or enemy_timestamp > prev_enemy_timestamp:
            prev_player_timestamp = player_timestamp
            prev_enemy_timestamp = enemy_timestamp
            checkpoint.load(
                run_dir, prefix='best',
                player_policy_net=player_policy_net,
                enemy_policy_net=enemy_policy_net
            )

        for event in pygame.event.get():

            actions = []

            if event.type == pygame.KEYDOWN:
                enemy_action = Action(enemy_policy(state))
                if event.key == pygame.K_a:
                    actions = [Action.BACKWARD, enemy_action]
                elif event.key == pygame.K_d:
                    actions = [Action.FORWARD, enemy_action]
                elif event.key == pygame.K_SPACE:
                    actions = [Action.ATTACK, enemy_action]
                elif event.key == pygame.K_s:
                    actions = [Action.NOOP, enemy_action]
                elif event.key == pygame.K_t:
                    print([[s[0].name, s[1].name] for s in trajectory])
                else:
                    break

            if event.type == pygame.QUIT:
                running = False

        if args.auto:
            actions = [Action(player_policy(state)), Action(enemy_policy(state))]

        if len(actions) == 2:
            print([a.name for a in actions])
            trajectory += [actions]
            state, reward, done, trunc, info = env.step(actions)
            print(state, reward, done)

            rgb = env.render()

        if done:
            print([[s[0].name, s[1].name] for s in trajectory])
            state, info = env.reset()
            rgb = env.render()
            done = False
            trajectory = []

    pygame.quit()
