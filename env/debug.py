import gym
import numpy as np

"""
A set of super simple environments for testing RL code during development
"""


class StaticEnv:
    def __init__(self):
        self.len = 3
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        self.state += 1
        done = self.len - 1 == self.state
        return self.state, 0.0, done, {}


class DummyEnv(gym.Env):
    def __init__(self, trajectories):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=2.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.trajectories = trajectories
        self.t = 0
        self.i = 0

    def reset(self):
        self.i = 0
        return self.trajectories[self.t][0][self.i]

    def step(self, action):
        self.i += 1
        s, r, d, i = self.trajectories[self.t][self.i]
        if d:
            self.t += 1
        return s, r, d, i


class LinearEnv(gym.Env):
    """
    A continuous environment such that...

            T - 0 -------- 1.0 - T
    Reward  T - 0 -------- 1.0 - T

    Action is deterministic float, simply adds or subtracts from the state

    """

    def __init__(self, inital_state=0.0):
        """
        initial: state
        """
        super().__init__()
        self.initial_state = inital_state
        self.state = np.array([inital_state])
        self.observation_space = gym.spaces.Box(low=-1.0, high=2.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def reset(self):
        self.state[0] = np.array([self.initial_state])
        return self.state.copy()

    def step(self, action):
        self.state[0] += action
        return self.state.copy(), self.reward(), self.done(), {}

    def reward(self):
        return 1.0 if self.state[0].item() > 1.0 else 0.0

    def done(self):
        return self.state[0].item() < 0.0 or self.state[0].item() > 1.0

    def render(self, mode='human'):
        pass


def one_hot(state, size):
    s = np.zeros(size)
    s[state] = 1.0
    return s


class LineGrid(gym.Env):
    def __init__(self, initial_state, n_states, reward_map):
        super().__init__()
        self.initial_state = initial_state
        self.state = initial_state
        self.observation_space = gym.spaces.Discrete(n_states)
        self.action_space = gym.spaces.Discrete(2)
        self.action_map = [-1, 1]
        self.reward_map = reward_map
        self.terminal_map = {0: True, n_states-1: True}

    def d_s(self, s, a):
        return self.action_map[a]

    def reward(self, s):
        return self.reward_map.get(s, 0.0)

    def done(self, s):
        return self.terminal_map.get(s, False)

    def reset(self):
        self.state = self.initial_state
        return one_hot(self.state, self.observation_space.n)

    def step(self, action: int):
        self.state += self.d_s(self.state, action)
        return one_hot(self.state, self.observation_space.n), self.reward(self.state), self.done(self.state), {}

    def lookahead(self, state, action):
        next_state = np.argmax(state)
        next_state += self.d_s(state, action)
        return one_hot(next_state, self.observation_space.n), self.reward(next_state), self.done(next_state), {}

    def render(self, mode=None):
        print(one_hot(self.state, self.observation_space.n))


class Bandit(LineGrid):
    def __init__(self):
        """
        S : Start state
        T : Terminal state
        () : Reward
        [T(-1.0), S, T(1.0)]
        """
        super().__init__(initial_state=1, n_states=3, reward_map=dict([(0, -1.0), (2, 1.0)]))


class DelayedBandit(LineGrid):
    def __init__(self):
        """
        S : Start state
        T : Terminal state
        () : Reward
        [T(-1.0), E, E, S, E, E, T(1.0)]
        """
        super().__init__(initial_state=3, n_states=7, reward_map=dict([(0, -1.0), (6, 1.0)]))
