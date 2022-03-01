import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def print3D(points):
    xs = points[:,0]
    ys = points[:,1]
    zs = points[:,2]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    
          
    ax.plot(xs, ys, zs)
    plt.show()


if __name__=="__main__":
    ptlist=np.loadtxt("car.txt")

    print(ptlist)
    print3D(ptlist)