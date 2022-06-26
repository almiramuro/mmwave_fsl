import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pickle

from lib.plot import *
"""
    Outlier removal (cluster) -> Aggregate frames (delay) -> Cluster (cluster)

    to add: normalize() after outlier removal
"""
class Preprocessing:
    def __init__(self, contents, tar_frames):
        
        self.raw = contents          # dictionary from pkl file
        self.f = tar_frames
        
        self.clustered = dict()
        self.N = 0
        self.aggregated = dict()

        for _, pts in contents.items():
            self.N += len(pts)

        self.cluster(outlier=True)      # cluster and normalize
        self.decay()                    # decay
        self.cluster(e = 0.125,outlier = False)

        # self.norm_pts = self.normalize(pts)
        

    def cluster(self, e = 0.1, min_samp = 3, outlier = False):
        """
            input:              self.points    #ndarray (n x 3)
            output:             xyzc            #dictionary with key = color; value = coordinates
            ---with colors --> for realtime
        """
        group = self.raw if outlier else self.aggregated
        # print(group == self.aggregated)

        for key, pts in group.items():
            # print(key, np.shape(pts))
            db = DBSCAN(eps=e, min_samples=min_samp).fit(pts)
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

                xyz = pts[class_member_mask & core_samples_mask]
                
                xyz = np.vstack((xyz,pts[class_member_mask & ~core_samples_mask]))
                
                
                # normalize
                if(outlier):    # removal of outlier
                    xyzc[k] = [self.normalize(xyz), col]
                else:           # clustering in aggregated
                    xyzc[col] = xyz
            if(outlier):    # removal of outlier
                self.clustered[key] = xyzc
            else:           # clustering in aggregated
                self.aggregated[key] = xyzc
           


    # def drop_duplicates(stream):
    #     i = 0
    #     delKey = []
    #     k = list(stream.keys())
    #     similar = dict() 

    #     for l in range(len(k) - 1):
    #         if(stream[k[l]].all() == stream[k[l+1]].all()):
    #             if i not in similar.keys():
    #                 similar[i] = 1
    #             else:
    #                 similar[i] += 1
    #             delKey.append(k[l+1])
    #         else:
    #             i += 1
    #     print(similar)
    #     for key in delKey:
    #         stream.pop(key)
        
    #     return stream

    def normalize(self, xyz):
        
        # -------- Rotate --------


        # -------- Translate --------    
        # # Get centroid
        npts = len(xyz)
        centroid = tuple(np.sum(xyz[:,i])/npts for i in range(3))

        # Translate all points to make centroid (0,0,0)
        trans = np.array([[xyz[i][j] - centroid[j] for j in range(3)] for i in range(npts)])
    
        return trans

    def decay(self):
        """
            raw = dictionary
            self.N = int total # of points in a gesture
            f = int # of desired frames
        """    
        frame = 1          # start frame count
        
        o = 0
        while(len(self.aggregated) != self.f):
            toDel = []      # keys already passed to agg thus no longer needed in raw
            ptcount = int(self.N/self.f)
            # print('raw:',len(self.raw.items()))
            for t, clusts in self.clustered.items():
                # self.clustered --> dict with timestamps as keys
                #                    values with another dictionary with cluster num as keys
                #                                                      values = [np.array(xyz),tuple(c)]
                for n, xyzc in clusts.items():
                    if(len(xyzc[0]) <= ptcount):
                        if(frame not in self.aggregated): 
                            self.aggregated[frame] = xyzc[0]
                        else: 
                            self.aggregated[frame] = np.vstack(xyzc[0])
                            
                        ptcount -= len(xyzc[0])
                        # toDel.append(key)   # store used key to toDel array
                # 
            frame += 1
        print(self.aggregated.items())
            # delete already processed keys
            # for i in toDel:
            #     raw.pop(i)
        
        # show all generated aggregated frames
        # for frame, pt in agg.items():
        #     print('frame:', frame, 'pts:' , len(pt) , '\n',pt)
        #     plot3d(pt)      
        # return agg
    
    def plot_output(self):
        for key, xyzc in self.clustered.items():
            print('_:',key,'xyz:',len(xyzc))
            plot3d_col(xyzc)

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

    sign = Preprocessing(pm_contents, 5)
    sign.plot_output()

    # print('ORIG # OF frames:',len(pm_contents.items()))
    # c = 0

    # # outlier removal and translation
    # for key, pts in pm_contents.items():
    #     c += len(pts)
    #     pm_contents[key] = cluster(pts, e = 0.8, outlier=True)
    #     pm_contents[key] = normalize(pts)
    
    # print('total number of points all in all: ',c)

    # # aggregate frames
    # aggframes = decay(pm_contents, c,5)
    
    # # cluster
    # for _, xyz in aggframes.items():
    #     print('_:',_,'xyz:',len(xyz))
    #     clust = cluster(xyz, e = 0.125)
    #     plot3d_col(clust.items())