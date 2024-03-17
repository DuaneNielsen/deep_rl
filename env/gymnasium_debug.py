from typing import SupportsFloat, Any

import gymnasium as gym
import numpy as np
from math import floor
from matplotlib import pyplot as plt
from gymnasium.core import WrapperActType, ActType, ObsType, RenderFrame
from gymnasium.spaces import MultiDiscrete, Discrete


class TugOfWar(gym.Env):
    """
    A continuous environment such that...

    .. code-block ::

                T - 0 -------- 1.0 - T
        Reward  T - 0 -------- 1.0 - T

    Action is deterministic float, simply adds or subtracts from the state

    """

    def __init__(self, inital_state=0.5):
        """
        initial: state
        """
        super().__init__()
        self.initial_state = inital_state
        self.state = np.array([inital_state], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=2.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.state[0] = np.array([self.initial_state], dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action):
        self.state[0] += sum(action)
        return self.state.copy(), self.reward(), self.done(), False, {}

    def reward(self):
        return floor(self.state[0].item())

    def done(self):
        return not 0. < self.state[0].item() < 1.

    def render(self, mode='human'):
        return np.ones((2, 2, 3)).astype(np.uint8)

    def states(self):
        return np.linspace(self.observation_space.low, self.observation_space.high, 10)


class TugOfWarDiscrete(gym.ActionWrapper):

    def __init__(self, env):
        super(TugOfWarDiscrete, self).__init__(env)
        self.action_space = gym.spaces.Discrete(2)

    def action(self, action: WrapperActType) -> ActType:
        return [(a - 0.5) / 2 for a in action]


class IteratedRockPaperScissors(gym.Env):

    def __init__(self, max_iterations=3):
        self.max_iterations = max_iterations
        self.iteration = 0
        self.state = np.zeros(self.max_iterations * 2, dtype=np.uint8)
        self.win_matrix = np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0]
        ])
        self.observation_space = MultiDiscrete([5]*max_iterations*2, dtype=np.uint8, seed=42)
        self._states = np.stack([self.observation_space.sample() for _ in range(9)] + [np.array([4]*max_iterations*2)])
        self.action_space = Discrete(3, seed=42)

    def reset(self, seed=None, options=None):
        self.iteration = 0
        self.state = np.zeros(self.max_iterations * 2, dtype=np.uint8)
        return self.state.copy(), {}

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.state[self.iteration*2:self.iteration*2+2] = [a + 1 for a in action]
        reward = self.win_matrix[action[0], action[1]]
        self.iteration += 1
        done = self.iteration == self.max_iterations
        if done:
            self.state[:] = 4
        return self.state.copy(), reward, done, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        # placeholder
        return np.ones((2, 2, 3), dtype=np.uint8)

    def states(self):
        return self._states