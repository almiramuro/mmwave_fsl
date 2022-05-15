import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
from collections import deque
import time
import threading


filename = 'hello.pkl'
with open(filename, 'rb') as handle:
    data = pickle.load(handle)

# print(b['1651302264453'].shape)
# print(b.keys())

def update(timestamp):
    data = q.popleft()
    # q.append([np.random.uniform(low=-3, high=3, size=8), np.random.uniform(low=-3, high=3, size=8), np.random.uniform(low=-3, high=3, size=8)])

    xs_ = data[timestamp][:,0]
    ys_ = data[timestamp][:,1]
    zs_ = data[timestamp][:,2]
    graph._offsets3d = (xs_, ys_, zs_)
    title = ax.set_title('3D Test, time={}'.format(timestamp))

q = deque()

t = list(data.keys())
for s in t:
    q.append(data[s])

print(q)
# test = q.popleft()
# print(test)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# title = ax.set_title('3D Test')

# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')
# ax.set_xlim3d(-3, 3)
# ax.set_ylim3d(-3, 3)
# ax.set_zlim3d(-3, 3)

# points = data[t[0]]

# xs = points[:,0]
# ys = points[:,1]
# zs = points[:,2]

# graph = ax.scatter(xs, ys, zs)
# # ax.scatter(xs, ys, zs)

# fig.canvas.draw()
# plt.show(block=False)