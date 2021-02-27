import io
import time
from datetime import datetime
from math import floor
from statistics import mean

import imageio
import numpy as np
import torch
from collections import OrderedDict, deque, namedtuple

from matplotlib import pyplot as plt
from torchvision.io import write_video, write_jpeg, write_png
from pathlib import Path
import gym


class EnvObserver:
    def reset(self, state):
        """ called when environment reset"""
        pass

    def step(self, action, state, reward, done, info, **kwargs):
        """ called each environment step """
        pass


class StateCapture(EnvObserver):
    def __init__(self):
        self.trajectories = []
        self.traj = []
        self.index = []
        self.cursor = 0

    def reset(self, state):
        self.traj.append(state)
        self.index.append(self.cursor)
        self.cursor += 1

    def step(self, action, state, reward, done, info, **kwargs):
        self.traj.append(state)
        self.index.append(self.cursor)
        self.cursor += 1

        if done:
            self.done()

    def done(self):
        self.trajectories += [self.traj]
        self.traj = []


class Enricher:
    """
    Base class used to enrich data collected during run
    will be called after buffer operations are complete
    multiple enrichers will be called in order they were attached
    """
    def reset(self, buffer, state, **kwargs):
        pass

    def step(self, buffer, action, state, reward, done, info, **kwargs):
        pass


class Returns(Enricher):
    """
    An enricher that calculates total returns
    Returns are added to the info field
    for transition (s, i, a, s_p, r, d, i_p), return = transition.i['g']
    """
    def step(self, buffer, action, state, reward, done, info, **kwargs):
        if done:
            # terminal state returns are always 0
            g = 0
            info['g'] = 0.0
            for s, i, a, s_p, r, d, i_p in TrajectoryTransitionsReverse(buffer, buffer.trajectories[-1]):
                g += r
                i['g'] = g


class DiscountedReturns(Enricher):
    """
    Enriches the transitions with discounted returns
    Returns are added to the info field
    for transition (s, i, a, s_p, r, d, i_p), return = transition.i['g']
    """
    def __init__(self, discount=0.95):
        self.discount = discount

    def step(self, buffer, action, state, reward, done, info, **kwargs):
        if done:
            # terminal state returns are always 0
            g = 0.0
            info['g'] = 0.0
            # get the last trajectory and reverse iterate over transitions
            for s, i, a, s_p, r, d, i_p in TrajectoryTransitionsReverse(buffer, buffer.trajectories[-1]):
                g = r + g * self.discount
                i['g'] = g


Transition = namedtuple('Transition', ['s', 'i', 'a', 's_p', 'r', 'd', 'i_p'])
"""
Transition

Attributes:
    s: state
    i: info dict for state
    a: action
    s_p: state prime, the resultant state
    r: reward
    d: done
    i_p: info dict for state prime
"""


class ReplayBuffer(EnvObserver):
    def __init__(self):
        """
        Replay buffer

        Attributes:
        buffer          [a, s, r, d, i), () ...]
                        2 transition trajectory would be..
                        [(None, s, 0.0, False, {}), (a, s, r, d, i), (a, s, r, d, i)]
        trajectories    [(start, end), ...]
        transitions     a flat index into buffer that points to the head of each transition
                        ie: to retrieve the n'th transition state, action, state', reward, done, info
                        transition_index = transitions[n]
                        _, state, _, _, _ = buffer[transition_index]
                        action, state_prime, reward, done, info = buffer[transtion_index + 1]
        """
        self.buffer = []
        self.trajectories = []
        self.transitions = []
        self.traj_start = 0
        self.enrich = []

    def attach_enrichment(self, enricher):
        self.enrich.append(enricher)

    def clear(self):
        """ clears the buffer """
        self.buffer = []
        self.trajectories = []
        self.transitions = []
        self.traj_start = 0

    def reset(self, state, **kwargs):
        self.buffer.append((None, state, 0.0, False, {}))
        self.transitions.append(len(self.buffer) - 1)

        for enricher in self.enrich:
            enricher.reset(self, state, **kwargs)

    def step(self, action, state, reward, done, info, **kwargs):
        self.buffer.append((action, state, reward, done, info))

        if done:
            """ terminal state, trajectory is complete """
            self.trajectories.append((self.traj_start, len(self.buffer)))
            self.traj_start = len(self.buffer)
        else:
            """ if not terminal, then by definition, this will be a transition """
            self.transitions.append(len(self.buffer)-1)

        for enricher in self.enrich:
            enricher.step(self, action, state, reward, done, info, **kwargs)

    def __getitem__(self, item):
        item = self.transitions[item]
        _, s, _, _, i = self.buffer[item]
        a, s_p, r, d, i_p = self.buffer[item+1]
        return Transition(s, i, a, s_p, r, d, i_p)

    def __len__(self):
        if len(self.buffer) == 0:
            return 0
        _, _, _, done, _ = self.buffer[-1]
        """ if the final state is not done, then we are still writing """
        if not done:
            """ so we can't use the transition at the end yet"""
            return len(self.transitions) - 1
        return len(self.transitions)


class TrajectoryTransitions:
    """
    Iterates over a trajectory in the buffer, from start to end, given a start:end tuple

    eg: to iterate over the most recent trajectory

    ```
    trajectory = Transition(buffer, buffer.trajectories[-1])
    ```

    """
    def __init__(self, replay_buffer, trajectory_start_end_tuple):
        self.buffer = replay_buffer
        self.start = trajectory_start_end_tuple[0]
        self.end = trajectory_start_end_tuple[1]
        self.cursor = self.start

    def __next__(self):
        if self.cursor + 1 < self.end:
            _, s, _, _, i = self.buffer.buffer[self.cursor]
            a, s_p, r, d, i_p = self.buffer.buffer[self.cursor + 1]
            self.cursor += 1
            return s, i, a, s_p, r, d, i_p
        else:
            raise StopIteration

    def __iter__(self):
        return self


class TrajectoryTransitionsReverse:
    def __init__(self, replay_buffer, trajectory_start_end_tuple):
        self.buffer = replay_buffer
        self.start = trajectory_start_end_tuple[0]
        self.end = trajectory_start_end_tuple[1]
        self.cursor = self.end - 1

    def __next__(self):
        if self.cursor > self.start:
            _, s, _, _, i = self.buffer.buffer[self.cursor - 1]
            a, s_p, r, d, i_p = self.buffer.buffer[self.cursor]
            self.cursor -= 1
            return s, i, a, s_p, r, d, i_p
        else:
            raise StopIteration

    def __iter__(self):
        return self


class VideoCapture(EnvObserver):
    def __init__(self, directory):
        self.t = []
        self.directory = directory
        self.cap_id = 0

    def reset(self, state):
        self.t.append(state)

    def step(self, action, state, reward, done, info, **kwargs):
        self.t.append(state)

        if done:
            self.done()

    def done(self):
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        stream = torch.from_numpy(np.stack(self.t))
        write_video(f'{self.directory}/capture_{self.cap_id}.mp4', stream, 24.0)
        self.cap_id += 1


class JpegCapture(EnvObserver):
    def __init__(self, directory):
        self.t = []
        self.directory = directory
        self.cap_id = 0
        self.image_id = 0

    def reset(self, state):
        self.t.append(state)

    def step(self, action, state, reward, done, info, **kwargs):
        self.t.append(state)

        if done:
            self.done()

    def done(self):
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        stream = torch.from_numpy(np.stack(self.t))
        for image in stream:
            write_jpeg(image.permute(2, 0, 1), f'{self.directory}/{self.image_id}.jpg')
            self.image_id += 1


class PngCapture(EnvObserver):
    def __init__(self, directory):
        self.t = []
        self.directory = directory
        self.cap_id = 0
        self.image_id = 0

    def reset(self, state):
        self.t.append(state)

    def step(self, action, state, reward, done, info, **kwargs):
        self.t.append(state)

        if done:
            self.done()

    def done(self):
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        stream = torch.from_numpy(np.stack(self.t))
        for image in stream:
            write_png(image.permute(2, 0, 1), f'{self.directory}/{self.image_id}.png')
            self.image_id += 1


class StepFilter:
    """
    Step filters are used to preprocess steps before handing them to observers

    if you want to pre-process environment observations before passing to policy, use a gym.Wrapper
    """
    def __call__(self, action, state, reward, done, info, **kwargs):
        return action, state, reward, done, info, kwargs


class RewardFilter(StepFilter):
    def __init__(self, state_prepro, R, device):
        self.state_prepro = state_prepro
        self.R = R
        self.device = device

    def __call__(self, action, state, reward, done, info, **kwargs):
        r = self.R(self.state_prepro(state, self.device))
        kwargs['model_reward'] = r.item()
        return action, state, reward, done, info, kwargs


class SubjectWrapper(gym.Wrapper):
    """
    gym wrapper with pluggable observers

    to attach an observer implement EnvObserver interface and use attach()

    filters to process the steps are supported, and data enrichment is possible
    by adding to the kwargs dict
    """
    def __init__(self, env, seed=None, **kwargs):
        gym.Wrapper.__init__(self, env)
        self.kwargs = kwargs
        self.env = env
        if seed is not None:
            env.seed(seed)
        self.observers = OrderedDict()
        self.step_filters = OrderedDict()

    def attach_observer(self, name, observer):
        self.observers[name] = observer

    def detach_observer(self, name):
        del self.observers[name]

    def observer_reset(self, state):
        for name, observer in self.observers.items():
            observer.reset(state)

    def append_step_filter(self, name, filter):
        self.step_filters[name] = filter

    def observe_step(self, action, state, reward, done, info, **kwargs):
        for name, filter in self.step_filters.items():
            action, state, reward, done, info, kwargs = filter(action, state, reward, done, info, **kwargs)

        for name, observer in self.observers.items():
            observer.step(action, state, reward, done, info, **kwargs)

    def reset(self, **kwargs):
        state = self.env.reset()
        self.observer_reset(state)
        return state

    def step(self, action, **kwargs):
        state, reward, done, info = self.env.step(action, **kwargs)
        self.observe_step(action, state, reward, done, info, **kwargs)
        return state, reward, done, info


def render_env(env, render, delay):
    if render:
        env.render(render)
        time.sleep(delay)


def episode(env, policy, render=False, delay=0.01, **kwargs):
    """
    Runs one episode using the provided policy on the environment
    :param policy: takes state as input, must output an action runnable on the environment
    :param render: if True will call environments render function
    :param delay: rendering delay
    :param kwargs: kwargs will be passed to policy, environment step, and observers
    """
    with torch.no_grad():
        state, reward, done, info = env.reset(**kwargs), 0.0, False, {}
        action = policy(state, **kwargs)
        render_env(env, render, delay)
        while not done:
            state, reward, done, info = env.step(action, **kwargs)
            action = policy(state, **kwargs)
            render_env(env, render, delay)


class Plot(EnvObserver):
    """
    An Observer that will plot episode returns and lengths
    to use, use the buffer.Subject gym wrapper and attach, ie
    env = gym.make('Cartpole-v1')
    env = buffer.SubjectWrapper(env)
    plotter = Plot()
    env.attach("plotter", plotter)
    """
    def __init__(self, refresh_cooldown=1.0, history_length=None, blocksize=1):
        """

        :param refresh_cooldown: maximum refresh frequency
        :param history_length: amount of trajectory returns to buffer
        :param blocksize: combines episodes into blocks and plots the average result
        """
        self.cols = 4
        self.rows = 2
        plt.ion()
        self.fig = plt.figure(figsize=(8, 16))
        spec = plt.GridSpec(ncols=self.cols, nrows=self.rows, figure=self.fig)

        self.update_cooldown = Cooldown(secs=refresh_cooldown)
        self.blocksize = blocksize

        self.epi_reward_ax = self.fig.add_subplot(spec[0, 0:4])
        self.epi_reward = deque(maxlen=history_length)
        self.block_ave_reward = []

        self.epi_len_ax = self.fig.add_subplot(spec[1, 0:4])
        self.epi_len = deque(maxlen=history_length)
        self.block_ave_len = []

    def reset(self, state, **kwargs):
        self.epi_reward.append(0)
        self.epi_len.append(0)

    def step(self, action, state, reward, done, info, **kwargs):
        self.epi_reward[-1] += reward
        self.epi_len[-1] += 1

        if done and self.update_cooldown():

            if len(self.epi_len) % self.blocksize:
                n = len(self.epi_len) // self.blocksize
                start, end = n * self.blocksize, (n + 1) * self.blocksize
                self.block_ave_reward += [mean(list(self.epi_reward)[start:end])]
                self.block_ave_len += [mean(list(self.epi_len)[start:end])]

            self.epi_reward_ax.clear()
            self.epi_reward_ax.plot(self.block_ave_reward)
            self.epi_len_ax.clear()
            self.epi_len_ax.plot(self.block_ave_len)
            plt.pause(0.001)

    def save(self):
        io_buf = io.BytesIO()
        self.fig.savefig(io_buf, format='png')
        self.vidstream.append(io_buf)

    def write_video(self):
        with imageio.get_writer('data/movie.mp4', mode='I', fps=0.8) as writer:
            for buffer in self.vidstream:
                buffer.seek(0)
                image = imageio.imread(buffer, 'png')
                writer.append_data(image)
                buffer.close()
        del self.vidstream
        self.vidstream = deque(maxlen=30000)


class Cooldown:
    def __init__(self, secs=None):
        """
        Cooldown timer. to use, just construct and call it with the number of seconds you want to wait
        default is 1 minute, first time it returns true
        """
        self.last_cooldown = 0
        self.default_cooldown = 60 if secs is None else secs

    def __call__(self, secs=None):
        secs = self.default_cooldown if secs is None else secs
        now = floor(datetime.now().timestamp())
        run_time = now - self.last_cooldown
        expired = run_time > secs
        if expired:
            self.last_cooldown = now
        return expired