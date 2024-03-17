import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import time

def moving_ave(ave, update):
    return ave * 0.9 + update * 0.1


class RLPlot(nn.Module):
    def __init__(self):
        num_rows, num_cols, figsize = 1, 1, (8, 6)
        super().__init__()
        plt.ion()
        self.fig = plt.figure(figsize=figsize)
        self.grid = plt.GridSpec(1, 1, wspace=0.4, hspace=0.5)
        self.num_axes = num_rows * num_cols
        self.ax = self.fig.add_subplot(self.grid[0, 0])
        self.fig.canvas.draw()
        self.value_min, self.value_max = -1, 1

    def update(self, value_f, policy, states, titles=None, xlabels=None, ylabels=None):
        self.ax.cla()
        param = next(value_f.parameters())
        states = torch.from_numpy(states).type(param.dtype).to(param.device)
        with torch.no_grad():
            values = value_f(states).cpu().numpy()
            self.value_min, self.value_max = moving_ave(self.value_min, values.min()), moving_ave(self.value_max, values.max())
            actions = policy(states).probs.argmax(1).cpu().numpy()
            self.ax.scatter(list(range(states.size(0))), values)
            self.ax.set_title('value')
            self.ax.set_xlabel('states')
            self.ax.set_ylim(self.value_min - abs(0.05 * self.value_min), self.value_max + abs(0.05 * self.value_max))

            # for i in range(self.num_axes):
            #     ax = self.fig.add_subplot(self.grid[i])
            #     self.axes.append(ax)
            #     if i < len(values):
            #         ax.hist(values[i])
            #         ax.set_title(titles[i] if titles else f"Plot {i+1}")
            #         ax.set_xlabel(xlabels[i] if xlabels else "X")
            #         ax.set_ylabel(ylabels[i] if ylabels else "Y")
            #     else:
            #         ax.axis('off')

    def draw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
