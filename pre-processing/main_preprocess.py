import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pickle
import pandas as pd
import os

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
        input:
            raw = dictionary
            k = int total # of points in a gesture
            f = int # of desired frames
        output:
            agg = dictionary with frame number (int) as key and xyz points (2d (n x 3) np array) as value 
    """    
    fcount = [i for i in range(1, f + 1)]
    pts_perframe = int(k/f)
    agg = {frame: np.zeros((1,3)) for frame in fcount}
    
    start = 0
    frame = 1
    agg[frame] = np.delete(agg[frame], 0, axis = 0)
    
    for key, pts in raw.items():    
        for pt in pts:
            agg[frame] = np.vstack((agg[frame], pt))
            if(len(agg[frame]) == pts_perframe and frame < f): 
                frame += 1
                agg[frame] = np.delete(agg[frame], 0, axis = 0)
            elif(len(agg[frame]) == pts_perframe and frame == f):
                return agg


def createMultiview(_3dframe):
    """
        input:
            _3dframe: dictionary 
                        >> key: cluster, value: 2d np array  (nx3)
        output:
            xyframes,yzframes,xzframes: 3 dictionaries 
                        >> per dictionary: key is cluster, value 2d np array (nx2)
    """
    xyframes,yzframes,xzframes = dict(), dict(), dict()
    for cluster, xyz in _3dframe.items():
        xy, yz, xz = [], [], []
        for X,Y,Z in xyz:
            xy.append([X,Y])
            yz.append([Y,Z])
            xz.append([X,Z])
        xyframes[cluster] = np.array(xy)
        yzframes[cluster] = np.array(yz)
        xzframes[cluster] = np.array(xz)

    # print(yzframes)
    return xyframes,yzframes,xzframes

def preprocess(filename, f):
    
    # Input handling
    """
        filename = string: <name>_<gloss>_<num>.pkl
        f = number of desired aggregated frames
    """

    #get gloss
    gloss = filename.split('_')[1].upper()

    with open(filename,"rb") as pm_data:
        pm_contents = pickle.load(pm_data,encoding ="bytes")

    # print('ORIG # OF frames:',len(pm_contents.items()))
    
    # Outlier Removal and Translation
    c = 0
    for key, pts in pm_contents.items():
        c += len(pts)
        pm_contents[key] = cluster(pts, e = 0.8, outlier=True)
        pm_contents[key] = normalize(pts)
    
    
    # Aggregate Frames
    # print(c, f)
    # print(int(c/f))

    aggframes = decay(pm_contents, c, f)

    
    
    # Cluster

    clustFrames = []                        # array to contain dictionaries
    for _, xyz in aggframes.items():        # iterated len(aggframes) times which is num of frames
        # print('_:',_,'xyz:',len(xyz))
        clust = cluster(xyz, e = 0.125, min_samp = 3, outlier=True)       # dictionary with key color c,and item of np array size n x 3 (pts)
        clustFrames.append(clust)
        # plot3d_col(clust.items())      # 1 frame 
    
    # Output handling 
    """
        CSV file 
        1 row = 1 frame data
        3 columns: xy, yz, xz
        per cell: 2d array (n x 3) 
    """
    data = {'xy': [], 'yz': [], 'xz': []} 
    for _3dframe in clustFrames:
        xyf,yzf,xzf = createMultiview(_3dframe)      # tuple containing 3 dictionaries xy, yz, xz 2dframes
        if(len(xyf.values()) == 0): continue
        data['xy'].append(np.concatenate(list(xyf.values())))
        data['yz'].append(np.concatenate(list(yzf.values())))
        data['xz'].append(np.concatenate(list(xzf.values())))

    print(len(data['xy'][0]),len(data['yz'][0]),len(data['xz'][0]))
    """ Save data into df and then save to csv file """
    df = pd.DataFrame(data)

    currDir = os.getcwd()
    subfolder = 'preprocessed_data'
    os.makedirs(os.path.join(currDir, subfolder), exist_ok=True)  
    newFile = filename[:-4] +'_processed.csv'
    df.to_csv(os.path.join(currDir, subfolder,newFile))  

    # trydf = pd.read_csv(os.path.join(currDir, subfolder,newFile))
    # print(trydf.xy[0])
    # print(len(trydf.xy[0]))

if __name__=="__main__":
        
    file_path = os.path.realpath(__file__)
    currDir = os.path.dirname(file_path)
    print(currDir)    
    
    os.chdir('..')
    rootDir = os.getcwd()

    if(currDir != rootDir):
        os.chdir(rootDir)
    

    dataDir = 'data'
    os.chdir(dataDir)

    raw_data = os.listdir()

    for file in raw_data:
        preprocess(file,40)