import pickle
import numpy as np
from matplotlib import pyplot as plt
from math import log10
import matplotlib.animation


filename = 'aaron_batangas_1'
with open(filename+'.pkl', 'rb') as handle:
    data = pickle.load(handle)

print(data)

t = list(data.keys())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')
ax.set_xlim3d(-3, 3)
ax.set_ylim3d(-3, 3)
ax.set_zlim3d(-3, 3)

points = data[t[0]]

xs = points[:,0]
ys = points[:,1]
zs = points[:,2]

graph = ax.scatter(xs, ys, zs)

# ani = matplotlib.animation.FuncAnimation(fig, update, t, interval=33, blit=False)

plt.show()
