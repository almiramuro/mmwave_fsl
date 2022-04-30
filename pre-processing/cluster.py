import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pickle

def cluster_wcolor(extractedPts, e = 0.1, min_samp = 3):
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
            col = (0, 0, 0, 1)

        class_member_mask = labels == k

        xyz = extractedPts[class_member_mask & core_samples_mask]
        
        xyz = np.vstack((xyz,extractedPts[class_member_mask & ~core_samples_mask]))
        xyzc[col] = xyz
    

def cluster(extractedPts, e = 0.1, min_samp = 3):
    """
        input:              extractedPts    #ndarray (n x 3)
        output:             xyzc            #dictionary with key = cluster; value = coordinates
        ---without colors
    """
    
    db = DBSCAN(eps=e, min_samples=min_samp).fit(extractedPts)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    labels = db.labels_
    unique_labels = set(labels)

    xyzc = dict()

    for k in unique_labels:
        class_member_mask = labels == k

        xyz = extractedPts[class_member_mask & core_samples_mask]
        
        xyz = np.vstack((xyz,extractedPts[class_member_mask & ~core_samples_mask]))
        xyzc[k] = xyz
    
    for k, pt in xyzc.items(): 
        print(k,':\n',pt,'\n---------------')
    return xyzc

if __name__=="__main__":

    with open('raw_2022-04-30_15-04-31.pkl',"rb") as pm_data:
        pm_contents = pickle.load(pm_data,encoding ="bytes")
    
    for ts, pts in pm_contents.items():
        print('time: ',ts)
        cluster(pts, e = 0.1)