import io
from collections import deque
from datetime import datetime
from math import floor
from statistics import mean

import gym
import imageio
from matplotlib import pyplot as plt


class Plot(gym.Wrapper):
    """
    An Observer that will plot episode returns and lengths
    to use, use the buffer.Subject gym wrapper and attach, ie
    env = gym.make('Cartpole-v1')
    env = buffer.SubjectWrapper(env)
    plotter = Plot()
    env.attach("plotter", plotter)
    """

    def __init__(self, env, refresh_cooldown=1.0, history_length=None, blocksize=1):
        """

        :param refresh_cooldown: maximum refresh frequency
        :param history_length: amount of trajectory returns to buffer
        :param blocksize: combines episodes into blocks and plots the average result
        """

        super().__init__(env)

        self.cols = 4
        self.rows = 2
        plt.ion()
        self.fig = plt.figure(figsize=(8, 16))
        if env.unwrapped.spec is not None:
            self.fig.suptitle(env.unwrapped.spec.id)
        else:
            self.fig.suptitle('set env.spec.id to show title')
        spec = plt.GridSpec(ncols=self.cols, nrows=self.rows, figure=self.fig)

        self.update_cooldown = Cooldown(secs=refresh_cooldown)
        self.blocksize = blocksize

        self.total_steps = 0
        self.total_step_tracker = []

        self.epi_reward_ax = self.fig.add_subplot(spec[0, 0:4])
        self.epi_reward = deque(maxlen=history_length)
        self.block_ave_reward = []

        self.epi_len_ax = self.fig.add_subplot(spec[1, 0:4])
        self.epi_len = deque(maxlen=history_length)
        self.block_ave_len = []

    def reset(self):
        self.epi_reward.append(0)
        self.epi_len.append(0)
        return self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.epi_reward[-1] += reward
        self.epi_len[-1] += 1
        self.total_steps += 1

        if done:
            """ calculate mean over prev block"""
            if len(self.epi_len) % self.blocksize == 0:
                n = len(self.epi_len) // self.blocksize
                start, end = (n - 1) * self.blocksize, n * self.blocksize
                self.block_ave_reward += [mean(list(self.epi_reward)[start:end])]
                self.block_ave_len += [mean(list(self.epi_len)[start:end])]
                self.total_step_tracker += [self.total_steps]

        if self.update_cooldown():

            self.epi_reward_ax.clear()
            self.epi_reward_ax.set_title('average reward per episode')
            self.epi_reward_ax.set_xlabel('steps')
            self.epi_reward_ax.plot(self.total_step_tracker, self.block_ave_reward)

            self.epi_len_ax.clear()
            self.epi_len_ax.set_title('average episode length')
            self.epi_len_ax.set_xlabel('steps')
            self.epi_len_ax.plot(self.total_step_tracker, self.block_ave_len)

            plt.pause(0.001)
        return state, reward, done, info

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