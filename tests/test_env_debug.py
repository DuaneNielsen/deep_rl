import buffer as bf
import torch
from torch import nn
import numpy as np

import driver
import observer
from algos import reinforce
import gym
from env import debug
import pytest


def test_linear_env():
    env = debug.LinearEnv()

    env.reset()
    assert env.state[0] == 0.0

    state, reward, done, info = env.step(0.5)
    assert state[0] == 0.5
    assert done is False
    assert reward == 0.0

    state, reward, done, info = env.step(0.5)
    assert state[0] == 1.0
    assert done is False
    assert reward == 0.0

    state, reward, done, info = env.step(0.1)
    assert state[0] == 1.1
    assert done is True
    assert reward == 1.0

    env.reset()
    state, reward, done, info = env.step(-0.1)
    assert state[0] == -0.1
    assert done is True
    assert reward == 0.0

    env.reset()
    state, reward, done, info = env.step(0.0)
    assert state[0] == 0.0
    assert done is False
    assert reward == 0.0

    def policy(state):
        dist = torch.distributions.normal.Normal(0, 0.5)
        return dist.rsample()

    env, buffer = bf.wrap(env)
    for i in range(5):
        driver.episode(env, policy)

    for start, end in buffer.trajectories:
        for transition in buffer.buffer[start:end]:
            state, action, reward, done, info = transition


@pytest.mark.skip(reason="requires tuning")
def test_REINFORCE():
    env = debug.LinearEnv(inital_state=0.1)
    env, buffer = bf.wrap(env)
    buffer.enrich(bf.DiscountedReturns())

    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([0.0]))
            self.b = nn.Parameter(torch.tensor([0.0]))

        def forward(self, state):
            loc = state * self.w + self.b
            return torch.distributions.Normal(loc=loc, scale=0.1)

    policy_net = PolicyNet().double()
    optim = torch.optim.SGD(policy_net.parameters(), lr=1e-3)

    def policy(state):
        state = torch.from_numpy(state)
        action = policy_net(state)
        return action.rsample().item()

    last_reward = 0
    for epoch in range(2000):
        for ep in range(32):
            driver.episode(env, policy)

        reward = 0
        for s, a, s_prime, r, d in buffer:
            reward += r
        last_reward = reward
        print(reward)

        reinforce.train(buffer, policy_net, optim, dtype=torch.double)

        buffer.clear()

    assert last_reward >= 14.0


def test_bandit():
    env = gym.make('Bandit-v1')

    state = env.reset()
    assert np.allclose(state, np.array([0.0, 1.0, 0.0]))

    state = env.reset()
    state, reward, done, info = env.step(0)
    assert np.allclose(state, np.array([1.0, 0.0, 0.0]))
    assert reward == -1.0
    assert done

    state = env.reset()
    state, reward, done, info = env.step(1)
    assert np.allclose(state, np.array([0.0, 0.0, 1.0]))
    assert reward == 1.0
    assert done


def test_delayed_bandit():
    env = gym.make('DelayedBandit-v1')

    state = env.reset()
    assert np.allclose(state, debug.one_hot(3, 7))

    state, reward, done, _ = env.step(1)
    assert np.allclose(state, debug.one_hot(4, 7))
    assert reward == 0.0
    assert not done

    state, reward, done, _ = env.step(1)
    assert np.allclose(state, debug.one_hot(5, 7))
    assert reward == 0.0
    assert not done

    state, reward, done, _ = env.step(0)
    assert np.allclose(state, debug.one_hot(4, 7))
    assert reward == 0.0
    assert not done

    state, reward, done, _ = env.step(1)
    assert np.allclose(state, debug.one_hot(5, 7))
    assert reward == 0.0
    assert not done

    next_state, reward, done, _ = env.lookahead(state, 0)
    assert np.allclose(next_state, debug.one_hot(4, 7))
    assert reward == 0.0
    assert not done

    next_state, reward, done, _ = env.lookahead(state, 1)
    assert np.allclose(next_state, debug.one_hot(6, 7))
    assert reward == 1.0
    assert done

    state, reward, done, _ = env.step(1)
    assert np.allclose(state, debug.one_hot(6, 7))
    assert reward == 1.0
    assert done

    state = env.reset()
    assert np.allclose(state, debug.one_hot(3, 7))

    state, reward, done, _ = env.step(0)
    assert np.allclose(state, debug.one_hot(2, 7))
    assert reward == 0.0
    assert not done

    state, reward, done, _ = env.step(0)
    assert np.allclose(state, debug.one_hot(1, 7))
    assert reward == 0.0
    assert not done

    state, reward, done, _ = env.step(1)
    assert np.allclose(state, debug.one_hot(2, 7))
    assert reward == 0.0
    assert not done

    state, reward, done, _ = env.step(0)
    assert np.allclose(state, debug.one_hot(1, 7))
    assert reward == 0.0
    assert not done

    next_state, reward, done, _ = env.lookahead(state, 0)
    assert np.allclose(next_state, debug.one_hot(0, 7))
    assert reward == -1.0
    assert done

    next_state, reward, done, _ = env.lookahead(state, 1)
    assert np.allclose(next_state, debug.one_hot(2, 7))
    assert reward == 0.0
    assert not done

    state, reward, done, _ = env.step(0)
    assert np.allclose(state, debug.one_hot(0, 7))
    assert reward == -1.0
    assert done