from pathlib import Path

import gym
import numpy as np
import torch
from torchvision.io import write_video, write_jpeg, write_png
from matplotlib import pyplot as plt
from collections import deque


class LiveMonitor(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.fig.canvas.draw()

    def reset(self):
        state = self.env.reset()
        self.ax.clear()
        self.ax.imshow(state)
        self.fig.canvas.draw()
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.ax.clear()
        self.ax.imshow(state)
        self.fig.canvas.draw()
        return state, reward, done, info


class VideoCapture(gym.Wrapper):
    """  Gym wrapper to create mp4 files from runs with visual output

        Args:
            env: environment to wrap
            directory: directory to write the files to
            freq: number of steps to to wait before capturing 1 episode
    """

    def __init__(self, env, directory, freq=10000, maxlen=3000):
        super().__init__(env)
        self.maxlen = maxlen
        self.t = deque(maxlen=maxlen)
        self.directory = directory
        self.cap_id = 1
        self.freq = freq
        self.total_steps = 0
        self.capturing = False

    def reset(self):
        """ wraps the gym reset call """
        state = self.env.reset()
        if self.total_steps > self.freq * self.cap_id:
            self.capturing = True
            self.t.append(state)
        return state

    def step(self, action):
        """ wraps the gym step call """
        state, reward, done, info = self.env.step(action)
        self.total_steps += 1
        if self.capturing:
            self.t.append(state)
            if done:
                self.capture()

        return state, reward, done, info

    def capture(self):
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        stream = torch.from_numpy(np.stack(self.t))
        write_video(f'{self.directory}/capture_{self.cap_id}.mp4', stream, 24.0)
        del self.t
        self.t = deque(maxlen=self.maxlen)
        self.cap_id += 1
        self.capturing = False


class JpegCapture(gym.Wrapper):
    """  Gym wrapper to create jpeg files from runs with visual output

        Args:
            env: environment to wrap
            directory: directory to write the files to
    """
    def __init__(self, env, directory):
        super().__init__(env)
        self.t = []
        self.directory = directory
        self.cap_id = 0
        self.image_id = 0

    def reset(self):
        """ wraps the gym reset call """
        state = self.env.reset()
        self.t.append(state)
        return state

    def step(self, action):
        """ wraps the gym step call """
        state, reward, done, info = self.env.step(action)
        self.t.append(state)

        if done:
            self.done()

        return state, reward, done, info

    def done(self):
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        stream = torch.from_numpy(np.stack(self.t))
        for image in stream:
            write_jpeg(image.permute(2, 0, 1), f'{self.directory}/{self.image_id}.jpg')
            self.image_id += 1


class PngCapture(gym.Wrapper):
    """  Gym wrapper to create png files from runs with visual output

        Args:
            env: environment to wrap
            directory: directory to write the files to
    """
    def __init__(self, env, directory):
        super().__init__(env)
        self.t = []
        self.directory = directory
        self.cap_id = 0
        self.image_id = 0

    def reset(self):
        """ wraps the gym step call """
        state = self.env.reset()
        self.t.append(state)
        return state

    def step(self, action):
        """ wraps the gym step call """
        state, reward, done, info = self.env.step(action)
        self.t.append(state)

        if done:
            self.done()

        return state, reward, done, info

    def done(self):
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        stream = torch.from_numpy(np.stack(self.t))
        for image in stream:
            write_png(image.permute(2, 0, 1), f'{self.directory}/{self.image_id}.png')
            self.image_id += 1