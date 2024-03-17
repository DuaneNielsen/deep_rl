import random

import gymnasium as gym
import torch.utils.data.sampler

import env
from random import randint
from torch.utils.data import DataLoader
import numpy as np


def step_env(env, policy, state=None, done=True, render=False):
    if state is None or done:
        state, info = env.reset()
    action = policy(state)
    state_p, reward, done, truncated, info = env.step(action)
    if render:
        env.render()
    return state, action, state_p, reward, done, truncated, info


def test_dataload():

    train_env = gym.make("IteratedRockPaperScissors-v1")
    buffer = []
    dl = DataLoader(buffer, batch_size=8, sampler=torch.utils.data.sampler.SequentialSampler(buffer))

    def player_explore_policy(state):
        return train_env.action_space.sample(), train_env.action_space.sample()

    (s_p, i), d = train_env.reset(), False
    for _ in range(8):
        s, a, s_p, r, d, _, i = step_env(train_env, player_explore_policy, s_p, d)
        buffer.append((s, a[0], s_p, r, d))

    state = [s for s, a, s_p, r, d in buffer]

    for s, a, s_p, r, d in dl:
        for i, s in enumerate(s):
            assert (s.cpu().numpy() == np.array(state[i])).all()


