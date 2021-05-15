import gym
import numpy as np
import torchvision.datasets as datasets
from torchvision.transforms.functional import resize
from matplotlib import pyplot as plt
import random
import torch

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
        return self.trajectories[self.t][self.i][0]

    def step(self, action):
        self.i += 1
        s, r, d, i = self.trajectories[self.t][self.i]
        if d:
            self.t += 1
        return s, r, d, i


class LinearEnv(gym.Env):
    """
    A continuous environment such that...

    .. code-block ::

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
    """
    Discrete linear environment in form

    [0, 0, 0, 1, 0, 0, 0]

    Args:
        inital_state: index of the state to start with, indexing is 0 .. n
        n_states: number of states 0 .. n
        reward_map: dictionary of states to that have rewards, keyed by integer of state

    going off the "end of the world" ends the episode

    """
    def __init__(self, initial_state, n_states, reward_map, obs_func=None):
        super().__init__()
        self.initial_state = initial_state
        self.state = initial_state
        self.observation_space = gym.spaces.Discrete(n_states)
        self.action_space = gym.spaces.Discrete(2)
        self.action_map = [-1, 1]
        self.reward_map = reward_map
        self.terminal_map = {0: True, n_states-1: True}
        self.obs_func = one_hot if obs_func is None else obs_func

    def d_s(self, s, a):
        """ given a state and action, returns the change in state """
        return self.action_map[a]

    def reward(self, s):
        """ reward for transitioning into a state """
        return self.reward_map.get(s, 0.0)

    def done(self, s):
        """ returns True if the state is terminal """
        return self.terminal_map.get(s, False)

    def reset(self):
        """ resets the environment to the initial state"""
        self.state = self.initial_state
        return one_hot(self.state, self.observation_space.n)

    def step(self, action: int):
        """ takes a step in the environment according to action """
        self.state += self.d_s(self.state, action)
        return one_hot(self.state, self.observation_space.n), self.reward(self.state), self.done(self.state), {}

    def lookahead(self, state, action):
        """ computes the next state, given a current state and action """
        next_state = np.argmax(state)
        next_state += self.d_s(state, action)
        return one_hot(next_state, self.observation_space.n), self.reward(next_state), self.done(next_state), {}

    def render(self, mode=None):
        """ prints the current state to the console """
        print(one_hot(self.state, self.observation_space.n))


class Bandit(LineGrid):
    """

    [T(-1.0), S, T(1.0)]

        S : Start state
        T : Terminal state
        (1.0) : Reward

    """
    def __init__(self):
        super().__init__(initial_state=1, n_states=3, reward_map=dict([(0, -1.0), (2, 1.0)]))


class DelayedBandit(LineGrid):
    """

    [T(-1.0), E, E, S, E, E, T(1.0)]

        S : Start state
        T : Terminal state
        (1.0) : Reward


    """
    def __init__(self):
        super().__init__(initial_state=3, n_states=7, reward_map=dict([(0, -1.0), (6, 1.0)]))


class MnistLineGrid(gym.Env):
    """

    [T(-1.0), S, T(1.0)]

        S : Start state
        T : Terminal state
        (1.0) : Reward

    """

    def __init__(self, initial_state, n_states, reward_map, easy=False,
                 mnist_dir='./data'):
        super().__init__()
        self.initial_state = initial_state
        self.state = initial_state
        self.observation_space = gym.spaces.Box(shape=(210, 160, 3), high=255, low=0, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self.action_map = [-1, 1]
        self.reward_map = reward_map
        self.terminal_map = {0: True, n_states-1: True}
        self.easy = easy

        self.mnist = datasets.MNIST(root=mnist_dir, train=False, download=False, transform=None)
        self.lookup = [[] for _ in range(10)]
        for data, target in zip(self.mnist.data, self.mnist.targets):
            self.lookup[target].append(data)

        self.fig = None
        self.ax = None

    def get_action_meanings(self):
        return ['LEFT', 'RIGHT']

    def mnist_obs(self, state):
        states = self.lookup[state]

        if self.easy:
            rand = 0
        else:
            rand = random.randint(0, len(states) - 1)

        raw = states[rand].unsqueeze(0).unsqueeze(0)
        enlarged = resize(raw, size=[210, 160]).squeeze()
        enlarged = enlarged.unsqueeze(2).repeat((1, 1, 3))
        return enlarged.numpy()

    def d_s(self, s, a):
        """ given a state and action, returns the change in state """
        return self.action_map[a.item()]

    def reward(self, s):
        """ reward for transitioning into a state """
        return self.reward_map.get(s, 0.0)

    def done(self, s):
        """ returns True if the state is terminal """
        return self.terminal_map.get(s, False)

    def reset(self):
        """ resets the environment to the initial state"""
        self.state = self.initial_state
        return self.mnist_obs(self.state)

    def step(self, action: int):
        """ takes a step in the environment according to action """
        self.state += self.d_s(self.state, action)
        return self.mnist_obs(self.state), self.reward(self.state), self.done(self.state), {}

    def render(self, mode=None):
        """ prints the current state to the console """
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(8, 8))
            spec = plt.GridSpec(ncols=1, nrows=1, figure=self.fig)
            self.ax = self.fig.add_subplot(spec[0, 0])
        self.ax.clear()
        self.ax.imshow(self.mnist_obs(self.state))
        self.fig.canvas.draw()


class MnistBandit(MnistLineGrid):
    """

    [T(-1.0), S, T(1.0)]

        S : Start state
        T : Terminal state
        (1.0) : Reward

    """
    def __init__(self, mnist_dir='./data', easy=False):
        super().__init__(initial_state=1, n_states=3, reward_map=dict([(0, -1.0), (2, 1.0)]),
                         mnist_dir=mnist_dir, easy=easy)


class MnistDelayedBandit(MnistLineGrid):
    """

    [T(-1.0), E, E, S, E, E, T(1.0)]

        S : Start state
        T : Terminal state
        (1.0) : Reward


    """
    def __init__(self, mnist_dir='./data', easy=False):
        super().__init__(initial_state=3, n_states=7, reward_map=dict([(0, -1.0), (6, 1.0)]),
                         mnist_dir=mnist_dir, easy=easy)


class MTGState:
    def __init__(self, pos=0, target=0):
        self.pos = pos
        self.target = target

    def __add__(self, other):
        self.pos += other.pos
        return self


class MnistTargetGrid(gym.Env):
    def __init__(self, initial_state, n_states,
                 mnist_dir='./data', easy=False):

        self.observation_space = gym.spaces.Box(shape=(210, 160, 3), high=255, low=0, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self.initial_state = initial_state
        self.easy = easy
        self.n = n_states
        self.action_map = [-1, 1]
        self.state = MTGState()
        self.mnist = datasets.MNIST(root=mnist_dir, train=False, download=False, transform=None)
        self.lookup = [[] for _ in range(10)]
        for data, target in zip(self.mnist.data, self.mnist.targets):
            self.lookup[target].append(data)

        self.fig = None
        self.ax = None

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
        return seed

    def get_action_meanings(self):
        return ['LEFT', 'RIGHT']

    def reward(self, s):
        if self.state.pos == self.state.target:
            self.state.target = random.randint(1, self.n-2)
            return 1.0
        else:
            return 0.0

    def d_s(self, state, a):
        return MTGState(pos=self.action_map[a])

    def get_image(self, i):
        images = self.lookup[i]
        if self.easy:
            rand = 0
        else:
            rand = random.randint(0, len(images) - 1)
        return self.lookup[i][rand]

    def obs(self, state):
        target = self.get_image(state.target)
        pos = self.get_image(state.pos)
        raw = torch.cat((pos, target), dim=0)
        raw = raw.unsqueeze(0).unsqueeze(0)
        enlarged = resize(raw, size=[210, 160]).squeeze()
        enlarged = enlarged.unsqueeze(2).repeat((1, 1, 3))
        return enlarged.numpy()

    def done(self, s):
        return s.pos == 0 or s.pos == self.n - 1

    def reset(self):
        self.state = MTGState(pos=self.initial_state, target=random.randint(1, self.n-2))
        return self.obs(self.state)

    def step(self, a):
        self.state += self.d_s(self.state, a.item())
        obs = self.obs(self.state)
        reward = self.reward(self.state)
        done = self.done(self.state)
        return obs, reward, done, {}

    def render(self, mode=None):
        """ prints the current state to the console """
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(8, 8))
            spec = plt.GridSpec(ncols=1, nrows=1, figure=self.fig)
            self.ax = self.fig.add_subplot(spec[0, 0])
        self.ax.clear()
        self.ax.imshow(self.obs(self.state))
        self.fig.canvas.draw()
