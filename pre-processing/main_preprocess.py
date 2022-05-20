import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pickle

from lib.plot import *
"""
    Outlier removal (cluster) -> Aggregate frames (delay) -> Cluster (cluster)

    to add: normalize() after outlier removal
"""

def cluster(extractedPts, e = 0.5, min_samp = 10, outlier = False):
    """
        input:              extractedPts    #ndarray (n x 3)
        output:             xyzc            #dictionary with key = color; value = coordinates
        ---with colors --> for realtime
    """

    db = DBSCAN(eps=e, min_samples=min_samp).fit(extractedPts)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    labels = db.labels_
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    xyzc = dict()

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            if(not outlier): col = (0, 0, 0, 1)
            continue
            

        class_member_mask = labels == k

        xyz = extractedPts[class_member_mask & core_samples_mask]
        
        xyz = np.vstack((xyz,extractedPts[class_member_mask & ~core_samples_mask]))
        if(outlier):
            i = k
        else: 
            i = col
        xyzc[i] = xyz
        
    return xyzc


def drop_duplicates(stream):
    i = 0
    delKey = []
    k = list(stream.keys())
    similar = dict() 

    for l in range(len(k) - 1):
        if(stream[k[l]].all() == stream[k[l+1]].all()):
            if i not in similar.keys():
                similar[i] = 1
            else:
                similar[i] += 1
            delKey.append(k[l+1])
        else:
            i += 1
    print(similar)
    for key in delKey:
        stream.pop(key)
    
    return stream

def normalize(xyz):
    """
        xyz = np array size n x 3
        get centroid 
        get centroid angle wrt boresight angle
        reference distance = 1.5m
        
    """
    
    # -------- Rotate --------


    # -------- Translate --------
    # Get centroid
    xyzsum = tuple(np.sum(xyz[:,i]) for i in range(3))
    npts = len(xyz)
    centroid = tuple(xyzsum[i]/npts for i in range(3))

    # Translate all points to make centroid (0,0,0)
    trans = np.array([[xyz[i][j] - centroid[j] for j in range(3)] for i in range(npts)])

    # Rotate along the z axis to face the sensor
    # norm = 0

    return trans

def decay(raw, k,f):
    """
        raw = dictionary
        k = int total # of points in a gesture
        f = int # of desired frames
    """    
    fcount = 1          # start frame count
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

    with open('hello3.pkl',"rb") as pm_data:
        pm_contents = pickle.load(pm_data,encoding ="bytes")
        
    # print(len(pm_contents))
    # # nodupes = drop_duplicates(pm_contents)
    # print(list(pm_contents.items())[0][1])
    # for _, pts in pm_contents.items():
    #     print(normalize(pts))
    #     break

    """divider only"""

    print('ORIG # OF frames:',len(pm_contents.items()))
    c = 0

    # outlier removal and translation
    for key, pts in pm_contents.items():
        c += len(pts)
        pm_contents[key] = cluster(pts, e = 0.8, outlier=True)
        pm_contents[key] = normalize(pts)
    
    print('total number of points all in all: ',c)

    # aggregate frames
    aggframes = decay(pm_contents, c,5)
    
    # cluster
    for _, xyz in aggframes.items():
        print('_:',_,'xyz:',len(xyz))
        clust = cluster(xyz, e = 0.125, min_samp = 3)
        plot3d_col(clust.items())