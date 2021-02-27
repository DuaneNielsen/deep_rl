import gym
import numpy as np
import gym.spaces as spaces
from collections import deque
import gym.wrappers
import cv2


class AtariAriVector(gym.Wrapper):
    def __init__(self, env, labels):
        gym.Wrapper.__init__(self, env)
        self.labels = labels
        self.max = np.array([205, 205, 205, 205], dtype=np.float32)
        self.min = np.array([0, 0, 0, 0], dtype=np.float32)
        shape = (len(self.labels), )
        self.observation_space = spaces.Box(low=0, high=1.0, shape=shape, dtype=np.float32)

    def extract(self, info):
        vector = []
        labels = info['labels']
        for field in self.labels:
            vector.append(labels[field])
        return (np.array(vector, dtype=np.float32) - self.min) / self.max

    def step(self, action):
        s, r, d, i = self.env.step(action)
        s = self.extract(i)
        return s, r, d, i

    def reset(self):
        self.env.reset()
        s, r, d, i = self.env.step(0)
        return self.extract(i)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        '''
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        '''
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        '''Do no-op action for a number of steps in [1, noop_max].'''
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        '''Take action on reset for environments that are fixed until firing.'''
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    '''OpenAI max-skipframe wrapper used for a NoFrameskip env'''

    def __init__(self, env, skip=4):
        '''Return only every `skip`-th frame'''
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        '''Repeat action, sum reward, and max over last observations.'''
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame doesn't matter
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class LazyFrames(object):
    def __init__(self, frames, frame_op='stack'):
        '''
        Wrapper to stack or concat frames by keeping unique soft reference insted of copies of data.
        So this should only be converted to numpy array before being passed to the model.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay buffers.
        @param str:frame_op 'stack' or 'concat'
        '''
        self._frames = frames
        self._out = None
        if frame_op == 'stack':
            self._frame_op = np.stack
        elif frame_op == 'concat':
            self._frame_op = np.concatenate
        else:
            raise ValueError('frame_op not recognized for LazyFrames. Choose from "stack", "concat"')

    def _force(self):
        if self._out is None:
            self._out = self._frame_op(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def astype(self, dtype):
        '''To prevent state.astype(np.float16) breaking on LazyFrames'''
        return self


class FrameStack(gym.Wrapper):
    def __init__(self, env, frame_op, frame_op_len):
        '''
        Stack/concat last k frames. Returns lazy array, which is much more memory efficient.
        @param str:frame_op 'concat' or 'stack'. Note: use concat for image since the shape is (1, 84, 84) concat-able.
        @param int:frame_op_len The number of frames to keep for frame_op
        '''
        gym.Wrapper.__init__(self, env)
        self.frame_op = frame_op
        self.frame_op_len = frame_op_len
        self.frames = deque([], maxlen=self.frame_op_len)
        self.rews = deque([], maxlen=self.frame_op_len)
        old_shape = env.observation_space.shape
        if self.frame_op == 'concat':  # concat multiplies first dim
            shape = (self.frame_op_len * old_shape[0],) + old_shape[1:]
        elif self.frame_op == 'stack':  # stack creates new dim
            shape = (self.frame_op_len,) + old_shape
        else:
            raise ValueError('frame_op not recognized for FrameStack. Choose from "stack", "concat".')
        self.observation_space = spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=shape, dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.frame_op_len):
            self.frames.append(ob.astype(np.float16))
            self.rews.append(0.0)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob.astype(np.float16))
        self.rews.append(reward)
        return self._get_ob(), self._get_rew(), done, info

    def _get_ob(self):
        assert len(self.frames) == self.frame_op_len
        return LazyFrames(list(self.frames), self.frame_op)

    def _get_rew(self):
        assert len(self.rews) == self.frame_op_len
        return LazyFrames(list(self.rews), self.frame_op)


class ActionBranches(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        savegame = self.env.clone_full_state()
        alternates = []
        for a in range(self.action_space.n):
            self.env.restore_full_state(savegame)
            state, rew, done, _ = self.env.step(a)
            alternates.append((state, rew, done))
        self.env.restore_full_state(savegame)
        state, rew, done, info = self.env.step(action)
        info['alternates'] = alternates
        return state, rew, done, info


class ConcatPrev(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_obs = None
        #shape = self.unwrapped.observation_space.shape[0] * 2
        low = np.concatenate((self.unwrapped.observation_space.low, self.unwrapped.observation_space.low))
        high = np.concatenate((self.unwrapped.observation_space.high, self.unwrapped.observation_space.high))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        self.prev_obs = self.env.reset()
        return np.concatenate((self.prev_obs, self.prev_obs))

    def step(self, action):
        raw_state, rew, done, info = self.env.step(action)
        state = np.concatenate((raw_state, self.prev_obs))
        self.prev_obs = raw_state
        return state, rew, done, info


class ConcatDeltaPrev(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_obs = None
        #shape = self.unwrapped.observation_space.shape[0] * 2
        low = np.concatenate((self.unwrapped.observation_space.low, self.unwrapped.observation_space.low))
        high = np.concatenate((self.unwrapped.observation_space.high, self.unwrapped.observation_space.high))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        self.prev_obs = self.env.reset()
        return np.concatenate((self.prev_obs, self.prev_obs - self.prev_obs))

    def step(self, action):
        raw_state, rew, done, info = self.env.step(action)
        state = np.concatenate((raw_state, raw_state - self.prev_obs))
        self.prev_obs = raw_state
        return state, rew, done, info


class AddDoneToState(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        low = np.concatenate((self.unwrapped.observation_space.low, np.zeros(1)))
        high = np.concatenate((self.unwrapped.observation_space.high, np.ones(1)))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        state = self.env.reset()
        return np.concatenate((state, np.ones(1, dtype=state.dtype)), axis=0)

    def step(self, action):
        raw_state, rew, done, info = self.env.step(action)
        state = np.concatenate((raw_state, np.full(1, fill_value=done, dtype=raw_state.dtype)), axis=0)
        return state, rew, done, info


class RewardOneIfNotDone(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        state, rew, done, info = self.env.step(action)
        reward = 0 if done else 1.0
        return state, reward, done, info


class Normalizer:
    def __init__(self, min, max):
        self.min, self.max = min, max

    def __call__(self, state):
        return (state - self.min) / (self.max - self.min)


class RewardCountLimit(gym.Wrapper):
    def __init__(self, env, max_reward_count=None):
        """
        Returns done once a number of nonzero rewards have been received
        :param env: the env to wrap
        :param max_reward_count: the
        """
        super().__init__(env)
        self.max_reward_count = max_reward_count
        self.reward_count = 0

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self.reward_count += 0 if reward == 0 else 1
        if self.reward_count >= self.max_reward_count:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        self.reward_count = 0
        return self.env.reset(**kwargs)


class RemapActions(gym.ActionWrapper):
    def __init__(self, env, remap):
        """

        :param env:
        :param remap: array of actions to remap
        """
        super().__init__(env)
        self.remap = remap
        self.action_space = gym.spaces.Discrete(len(remap))

    def action(self, action):
        return self.remap[action]


class ClipState2D(gym.ObservationWrapper):
    def __init__(self, env, x, y, h, w):
        super().__init__(env)
        self.x = x
        self.y = y
        self.h = h
        self.w = w

        num_colors = env.observation_space.shape[2]

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.h, self.w, num_colors),
            dtype=np.uint8,
        )
        self.observation_space = new_space

    def observation(self, observation):
        return observation[self.y:self.y+self.h, self.x:self.x+self.w]


class ApplyFunc(gym.ObservationWrapper):
    def __init__(self, env, func, output_observation_space):
        """
        Apply the given function to the observation and return the output
        :param env: environment
        :param func: function(observation) -> output
        :param output_observation_space: description of the output space, eg:
            new_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.h, self.w, num_colors),
                dtype=np.uint8,
            )
        """
        super().__init__(env)
        self.func = func
        self.observation_space = output_observation_space

    def observation(self, observation):
        return self.func(observation)


class Resize2D(gym.ObservationWrapper):
    """
    Apply the given function to the observation and return the output
    :param env: environment
    :param output_observation_space: gym.spaces.Box object
        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.h, self.w, num_colors),
            dtype=np.uint8,
        )
    """

    def __init__(self, env, h, w, interpolation=cv2.INTER_LINEAR):
        super().__init__(env)
        num_colors = env.observation_space.shape[2]
        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(h, w, num_colors),
            dtype=np.uint8,
        )
        self.observation_space = new_space
        self.interpolation = interpolation

    def observation(self, observation):
        return cv2.resize(observation, dsize=self.observation_space.shape[0:2], interpolation=self.interpolation)