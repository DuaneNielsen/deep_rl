import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
import matplotlib.colors
from collections import deque
import numpy as np
from argparse import ArgumentParser
import buffer_h5 as b5
import cv2
import torch
from torchvision.utils import make_grid
from time import time


class Trakker():
    def __init__(self, ax, N, L):
        self.ax, self.N, self.L = ax, N, L
        self.my_cmap = matplotlib.colors.ListedColormap(['r', 'g', 'b'])
        # set the 'bad' values (nan) to be white and transparent
        self.my_cmap.set_bad(color='w', alpha=0)
        # draw the grid
        for x in range(N + 1):
            self.ax.axhline(x, lw=2, color='k', zorder=5)
            self.ax.axvline(x, lw=2, color='k', zorder=5)
        self.data = deque([np.ones(N) * np.nan for _ in range(self.L)], maxlen=self.N)
        self.color = {'r': 0, 'g': 1, 'b': 2}

    def update(self, discrete, color=1):
        one_hot = np.ones(self.N) * np.nan
        one_hot[discrete] = color
        self.data.append(one_hot)
        data = np.stack(self.data)
        self.ax.clear()
        self.ax.imshow(data, interpolation='none', cmap=self.my_cmap, extent=[0, self.N, 0, self.L], zorder=0)


class Viewer:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure()
        spec = GridSpec(ncols=4, nrows=1)
        self.state_ax = self.fig.add_subplot(spec[0, 0])
        self.next_state_ax = self.fig.add_subplot(spec[0, 1])
        self.action_ax = self.fig.add_subplot(spec[0, 2])
        self.terminal_ax = self.fig.add_subplot(spec[0, 3])
        self.action_trakker = Trakker(self.action_ax, 4, 5)
        self.terminal_trakker = Trakker(self.terminal_ax, 2, 5)
        self.fig.canvas.draw()
        plt.pause(0.2)

    def update(self, state, action, next_state, terminal):
        self.state_ax.clear()
        self.state_ax.imshow(state)
        if next_state:
            self.next_state_ax.clear()
            self.next_state_ax.imshow(next_state)
        self.action_trakker.update(action)
        self.terminal_trakker.update(1 if terminal else 0)
        self.fig.canvas.draw()
        plt.waitforbuttonpress()


if __name__ == '__main__':
    parser = ArgumentParser(description='configuration switches')
    parser.add_argument('mode', choices=['transition', 'grid', 'episodes'], default='transition')
    parser.add_argument('filename', type=str, default='buffer.h5')
    parser.add_argument('--gridsize', type=int, default=32)
    parser.add_argument('--start_at', type=int, default=0)
    parser.add_argument('--delay', type=int, default=20)
    args = parser.parse_args()

    buffer = b5.Buffer()
    buffer.load(args.filename)

    if args.mode == 'transition':
        viewer = Viewer()
        for i in range(buffer.steps):
            raw, a, d = buffer.n_gram(i, 1, ['raw', 'action', 'done'])
            viewer.update(raw[0], a[0], None, d[0])

    if args.mode == 'grid':
        for i in range(args.start_at//args.gridsize, buffer.steps//args.gridsize):
            offset = i * args.gridsize
            raw = buffer.f['/replay/raw'][offset:offset+args.gridsize]
            grid = make_grid(torch.from_numpy(raw).permute(0, 3, 1, 2))
            cv2.imshow('raw', grid.permute(1, 2, 0).numpy())
            cv2.waitKey(args.delay)

    if args.mode == 'episodes':

        assert args.start_at < buffer.num_episodes, f"--start_at must be smaller than number of episodes {buffer.num_episodes}"

        def episode(start_at=0):
            i = start_at
            while True:
                start = buffer.episodes[i]
                if i < buffer.num_episodes -1:
                    end = buffer.episodes[i + 1]
                    assert end - start > 0
                    yield [buffer.raw[start:end], 0]
                elif i == buffer.num_episodes-1:
                    end = buffer.steps
                    assert end - start > 0
                    i += 1
                    yield [buffer.raw[start:end], 0]
                else:
                    yield None
                i += 1

        frames = [[np.empty(0), 0] for _ in range(args.gridsize)]
        none_count = 0
        grid = None
        next_episode = episode(args.start_at)
        while True:

            start_t = time()

            for f in range(args.gridsize):
                if frames[f] is not None:
                    if frames[f][1] >= frames[f][0].shape[0]:
                        frames[f] = next(next_episode)

            grid = []
            for f in frames:
                if f is not None:
                    grid.append(f[0][f[1]])

            for f in range(args.gridsize):
                if frames[f] is not None:
                    frames[f][1] += 1
                else:
                    none_count += 1

            grid_t = time()

            if none_count == args.gridsize:
                exit()
            else:
                print(none_count)
                none_count = 0

            grid = np.stack(grid)
            grid = make_grid(torch.from_numpy(grid).permute(0, 3, 1, 2))
            cv2.imshow('raw', grid.permute(1, 2, 0).numpy())
            cv2.waitKey(args.delay)

            display_t = time()
            #print(f'grid_t: {grid_t-start_t} display_t: {display_t-grid_t} delete_t: {delete_t-display_t}')

