import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation


filename = 'raw_2022-05-15_12-30-49.pkl'
with open(filename, 'rb') as handle:
    data = pickle.load(handle)

# print(b['1651302264453'].shape)
# print(b.keys())

def update(timestamp):
    xs_ = data[timestamp][:,0]
    ys_ = data[timestamp][:,1]
    zs_ = data[timestamp][:,2]
    graph._offsets3d = (xs_, ys_, zs_)
    title = ax.set_title('3D Test, time={}'.format(timestamp))
    

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

ani = matplotlib.animation.FuncAnimation(fig, update, t, interval=33, blit=False)

plt.show()