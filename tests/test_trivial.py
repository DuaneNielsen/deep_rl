import buffer as run
import torch
from torch import nn
import numpy as np

from algos import reinforce
from env import trivial


def test_linear_env():
    env = trivial.LinearEnv()

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

    runner = run.SubjectWrapper(env)

    def policy(state):
        dist = torch.distributions.normal.Normal(0, 0.5)
        return dist.rsample()

    replay_buffer = run.ReplayBuffer()
    runner.attach_observer("replay_buffer", replay_buffer)
    for i in range(5):
        run.episode(runner, policy)

    for start, end in replay_buffer.trajectories:
        for transition in replay_buffer.buffer[start:end]:
            state, action, reward, done, info = transition


def test_REINFORCE():
    env = trivial.LinearEnv(inital_state=0.1)
    buffer = run.ReplayBuffer()
    buffer.attach_enrichment(run.DiscountedReturns())
    env = run.SubjectWrapper(env)
    env.attach_observer("replay_buffer", buffer)

    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([0.0]))
            self.b = nn.Parameter(torch.tensor([0.0]))

        def forward(self, state):
            loc = state * self.w + self.b
            return torch.distributions.Normal(loc=loc, scale=0.25)

    policy_net = PolicyNet().double()
    optim = torch.optim.SGD(policy_net.parameters(), lr=0.1)

    def policy(state):
        state = torch.from_numpy(state)
        action = policy_net(state)
        return action.rsample().item()

    last_reward = 0
    for epoch in range(16):
        for ep in range(16):
            run.episode(env, policy)

        reward = 0
        for s, i, a, s_prime, r, d, i_p in buffer:
            reward += r
        last_reward = reward
        print(reward)

        reinforce.train(buffer, policy_net, optim, dtype=torch.double)

        buffer.clear()

    assert last_reward >= 14.0


def test_bandit():
    env = trivial.Bandit()

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