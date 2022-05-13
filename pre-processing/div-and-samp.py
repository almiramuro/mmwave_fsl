import numpy as np
import matplotlib.pyplot as plt
import pickle
from cluster import *

"""
    frame divider and resampling (need kasi ng PointNet++, let's see sa CNN)

    30fps 
    time decay
    total # of frames = total # of timestamps (ask luis if pwede palitan yung format ng timestamps)
"""

def plot3d(pts):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.set_xlim3d(-2,2)
    ax.set_ylim3d(0,3)
    ax.set_zlim3d(-2,2)
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot3d_col(xyzc):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-2,2)
    ax.set_ylim3d(0,3)
    ax.set_zlim3d(-2,2)
    for col, pts in xyzc:        
        ax.scatter3D(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color = tuple(col))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def decay(raw, k,f):
    # k = total # of points in a gesture
    # f = # of desired frames
    
    """
        Plan: Reduce the number of points by squeezing them into frames
        Steps:
        1. get k/f = points per new frame 
        2. for every old frame (raw), create a new dictionary with key, item:
            key = frame#
            item = pts
            keep adding pts into the key group/frame until it reaches the value k/f
        3. drop frames if less than k/f pts
        4. Show 3D image of frames 
    """
    fcount = 1
    agg = dict()        # aggregated point clouds into f frames

    while(len(agg) != f):
        toDel = []      # keys already passed to agg thus no longer needed in raw
        ptcount = int(k/f)
        print('raw:',len(raw.items()))
        for key, pts in raw.items():
            if(len(pts) <= ptcount):
                if(fcount not in agg): 
                    agg[fcount] = pts
                else: 
                    agg[fcount] = np.append(agg[fcount],pts,axis=0)
                ptcount -= len(pts)
                toDel.append(key)   # store used key to toDel array
        print(ptcount)
        fcount += 1
        
        # delete already processed keys
        for i in toDel:
            raw.pop(i)
    
    # show all generated aggregated frames
    # for frame, pt in agg.items():
    #     print('frame:', frame, 'pts:' , len(pt) , '\n',pt)
    #     plot3d(pt)      
    return agg

if __name__=="__main__":

    with open('raw_2022-04-30_15-04-31.pkl',"rb") as pm_data:
        pm_contents = pickle.load(pm_data,encoding ="bytes")
    
    # print(pm_contents.items())
    aggframes = decay(pm_contents, 670,2)
    # print(aggframes)
    for _, xyz in aggframes.items():
        print('_:',_,'xyz:',xyz)
        clust = cluster_wcolor(xyz, e = 0.2)
        plot3d_col(clust.items())