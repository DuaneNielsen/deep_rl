import gym
import numpy as np


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
        self.observation_space = gym.spaces.Box(low=-1.0, high=2.0, shape=(1, ), dtype=np.float32)
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

    def __init__(self, inital_state=0.0):
        super().__init__()
        self.initial_state = inital_state
        self.state = np.array([inital_state])
        self.observation_space = gym.spaces.Box(low=-1.0, high=2.0, shape=(1, ), dtype=np.float32)
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