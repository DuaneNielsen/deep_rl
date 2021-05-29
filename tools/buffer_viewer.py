import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
import matplotlib.colors
from collections import deque
import numpy as np
from argparse import ArgumentParser
import buffer_h5 as b5


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
    parser.add_argument('--filename', type=str, default='buffer.h5')
    config = parser.parse_args()

    buffer = b5.Buffer()
    buffer.load(config.filename)
    viewer = Viewer()

    for i in range(buffer.steps):
        raw, a, d = buffer.n_gram(i, 1, ['raw', 'action', 'done'])
        viewer.update(raw[0], a[0], None, d[0])

