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
# print(data[t[0]][0][3])
# print(data[1652948719.6900349])

# for timestamp in t:
#     for point in data[timestamp]:
#         print(timestamp, 10*log10(point[3]))
        

# with open(filename+'.txt', 'w') as f:
#     for timestamp in t:
#         for point in data[timestamp]:
#             x = point[0]
#             y = point[1]
#             z = point[2]
#             intensity = "{0:.4f}".format(10*log10(point[3]))
#             time = int(timestamp*100)
#             f.write("x::{} y::{} z::{} intensity::{} time::{}".format(x, y, z, intensity, time))
#             f.write('\n')


# --- PLOT ---
# def update(timestamp):
#     xs_ = data[timestamp][:,0]
#     ys_ = data[timestamp][:,1]
#     zs_ = data[timestamp][:,2]
#     graph._offsets3d = (xs_, ys_, zs_)
#     title = ax.set_title('3D Test, time={}'.format(timestamp))
    


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
