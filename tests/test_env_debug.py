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
from matplotlib import pyplot as plt

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


def test_mnist_bandit():
    env = debug.MnistBandit(mnist_dir='./data')

    state = env.reset()
    assert np.allclose(state, env.mnist_obs(1))

    state = env.reset()
    state, reward, done, info = env.step(0)
    assert np.allclose(state, env.mnist_obs(0))
    assert reward == -1.0
    assert done

    state = env.reset()
    state, reward, done, info = env.step(1)
    assert np.allclose(state, env.mnist_obs(2))
    assert reward == 1.0
    assert done


def test_mnist_target_grid():
    # plt.ion()
    # fig = plt.figure(figsize=(8, 8))
    # spec = plt.GridSpec(ncols=1, nrows=2, figure=fig)
    # ax = fig.add_subplot(spec[0, 0])
    # ax2 = fig.add_subplot(spec[1, 0])
    env = debug.MnistTargetGrid(mnist_dir='./data', initial_state=1, n_states=3, easy=True)
    env.seed(1)
    state = env.reset()
    expected = env.obs(debug.MTGState(1, 0))
    # env.render()
    # ax.clear()
    # ax2.clear()
    # ax.imshow(expected)
    # ax2.imshow(state)
    # fig.canvas.draw()
    assert np.allclose(state, expected)

    state, reward, done, info = env.step(np.array([0]))
    expected = env.obs(debug.MTGState(0, 0))
    assert np.allclose(state, expected)
    assert reward == 1.0
    assert done == True


def test_mnist_target_grid_long():
    env = debug.MnistTargetGrid(mnist_dir='./data', initial_state=3, n_states=7, easy=True)
    env.seed(1)
    state = env.reset()
    expected = env.obs(debug.MTGState(3, 1))
    assert np.allclose(state, expected)

    state, reward, done, info = env.step(np.array([0]))
    expected = env.obs(debug.MTGState(2, 1))
    assert np.allclose(state, expected)
    assert reward == 0.0
    assert done == False

    state, reward, done, info = env.step(np.array([0]))
    expected = env.obs(debug.MTGState(1, 1))
    assert np.allclose(state, expected)
    assert reward == 1.0
    assert done == False

    state, reward, done, info = env.step(np.array([0]))
    expected = env.obs(debug.MTGState(0, 6))
    assert np.allclose(state, expected)
    assert reward == 0.0
    assert done == True