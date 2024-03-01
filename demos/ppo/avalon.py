from avalonsim import Action
import random
import gym
from models.mlp import MLP, ActionHead
import torch
import checkpoint
from algos.ppo import PPOWrapModel
from avalonsim.wrappers import NoTurnaroundWrapper

if __name__ == "__main__":

    env = gym.make('Avalon-v1')
    env = NoTurnaroundWrapper(env)
    hidden_dim = 128
    enemy_policy_net = MLP(
        in_features=env.observation_space.shape[0],
        hidden_dims=hidden_dim,
        head=ActionHead(
            hidden_dims=hidden_dim,
            actions=env.action_space.n,
            exploration_noise=0.)
    )
    enemy_policy_net = PPOWrapModel(enemy_policy_net)

    checkpoint.load('C:/Users/Cabron/PycharmProjects/deep_rl/baselines/ppo/runs/run_75', prefix='best', enemy_policy_net=enemy_policy_net)

    def enemy_policy(state):
        with torch.no_grad():
            state = torch.from_numpy(state).type(torch.float32).unsqueeze(0)
            return enemy_policy_net(state).sample().item()

    import pygame

    running = True

    state = env.reset()
    print(state)

    rgb = env.render(mode='human')
    random.seed(42)

    trajectory = []
    done = False

    while running:
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

            if len(actions) == 2:
                print([a.name for a in actions])
                trajectory += [actions]
                state, reward, done, info = env.step(actions)
                print(state, reward, done)

                rgb = env.render(mode='human')

            if done:
                print([[s[0].name, s[1].name] for s in trajectory])
                state = env.reset()
                rgb = env.render(mode='human')
                done = False
                trajectory = []

    pygame.quit()
