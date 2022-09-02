from matplotlib import pyplot as plt
import numpy as np

H, W = 11, 11


# convert co-ordinates to index
def idx(x, y):
    return x * H + y


# return co-ordinates for N, S, E, W triangle with respect to the center triangle
def north(x, y):
    return (x, y), (x + 1, y + 1), (x - 1, y + 1)


def east(x, y):
    return (x, y), (x + 1, y - 1), (x + 1, y + 1)


def south(x, y):
    return (x, y), (x - 1, y - 1), (x + 1, y - 1)


def west(x, y):
    return (x, y), (x - 1, y + 1), (x - 1, y - 1)


# compute all the center co-ordinates in the grid
c_x, c_y = np.meshgrid(np.arange(5), np.arange(5), indexing='ij')

# compute the all the triangles
triangles = []
colors = []
for c_x, c_y in zip(c_x.flatten(), c_y.flatten()):
    c_x, c_y = c_x * 2 + 1, c_y * 2 + 1

    triangles += [north(c_x, c_y)]
    colors += [1.0]

    triangles += [east(c_x, c_y)]
    colors += [0.0]

    triangles += [south(c_x, c_y)]
    colors += [1.0]

    triangles += [west(c_x, c_y)]
    colors += [0.0]

# convert triangles from xy to vertex index
triangles = [[idx(*xy) for xy in tri] for tri in triangles]

# vertices at grid co-ordinates in increments of 0.5
x, y = np.meshgrid(np.linspace(0, H//2, H), np.linspace(0, W//2, W), indexing='ij')

fig = plt.figure()
ax = plt.subplot()
ax.tripcolor(x.flatten(), y.flatten(), triangles, colors, cmap='summer')
plt.show()
