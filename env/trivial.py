import gym
import numpy as np
import math


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

            | - 0 -------- 1.0 |
    Reward  | - 0 -------- 1.0 |

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


def discrete_state(state, size):
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

    def reset(self):
        self.state = self.initial_state
        return discrete_state(self.state, self.observation_space.n)

    def step(self, action: int):
        self.state += self.action_map[action]
        reward = 0.0
        if self.state in self.reward_map:
            reward = self.reward_map[self.state]
        done = self.state == 0 or self.state == self.observation_space.n - 1
        return discrete_state(self.state, self.observation_space.n), reward, done, {}

    def render(self, mode=None):
        print(self.state)


class Bandit(LineGrid):
    def __init__(self):
        """
        S : Start state
        T : Terminal state
        () : Reward
        [T(-1.0), S, T(1.0)]
        """
        super().__init__(1, 3, dict([(0, -1.0), (2, 1.0)]))


class DelayedBandit(LineGrid):
    def __init__(self):
        """
        S : Start state
        T : Terminal state
        () : Reward
        [T(-1.0), E, E, S, E, E, T(1.0)]
        """
        super().__init__(3, 7, dict([(0, -1.0), (6, 1.0)]))
