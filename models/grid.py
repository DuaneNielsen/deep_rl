import matplotlib.pyplot as plt
import numpy as np


class GridPlot:
    def __init__(self, num_rows, num_cols, figsize=(8, 6)):
        self.fig = plt.figure(figsize=figsize)
        self.grid = plt.GridSpec(num_rows, num_cols, wspace=0.4, hspace=0.5)
        self.num_axes = num_rows * num_cols
        self.axes = []

    def update(self, data, titles=None, xlabels=None, ylabels=None):
        for i in range(self.num_axes):
            ax = self.fig.add_subplot(self.grid[i])
            self.axes.append(ax)
            if i < len(data):
                ax.plot(data[i])
                ax.set_title(titles[i] if titles else f"Plot {i+1}")
                ax.set_xlabel(xlabels[i] if xlabels else "X")
                ax.set_ylabel(ylabels[i] if ylabels else "Y")
            else:
                ax.axis('off')

    def show(self):
        plt.show()


if __name__ == '__main__':
    # Example usage
    num_rows, num_cols = 2, 3
    grid_plot = GridPlot(num_rows, num_cols)

    # Generate some example data
    data = [np.random.randn(100) for _ in range(num_rows * num_cols)]
    titles = [f"Plot {i + 1}" for i in range(num_rows * num_cols)]

    grid_plot.update(data, titles=titles)
    grid_plot.show()