from pathlib import Path

import gym
import numpy as np
import torch
from torchvision.io import write_video, write_jpeg, write_png


class VideoCapture(gym.Wrapper):
    def __init__(self, env, directory):
        super().__init__(env)
        self.t = []
        self.directory = directory
        self.cap_id = 0

    def reset(self):
        state = self.env.reset()
        self.t.append(state)
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.t.append(state)

        if done:
            self.done()

        return state, reward, done, info

    def done(self):
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        stream = torch.from_numpy(np.stack(self.t))
        write_video(f'{self.directory}/capture_{self.cap_id}.mp4', stream, 24.0)
        self.cap_id += 1


class JpegCapture(gym.Wrapper):
    def __init__(self, env, directory):
        super().__init__(env)
        self.t = []
        self.directory = directory
        self.cap_id = 0
        self.image_id = 0

    def reset(self):
        state = self.env.reset()
        self.t.append(state)
        return state

    def step(self, action):
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
    def __init__(self, env, directory):
        super().__init__(env)
        self.t = []
        self.directory = directory
        self.cap_id = 0
        self.image_id = 0

    def reset(self):
        state = self.env.reset()
        self.t.append(state)
        return state

    def step(self, action):
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