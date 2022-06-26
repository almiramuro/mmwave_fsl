from dl_model.model import wordNet
import numpy as np 
import torch.nn as nn
import torch
import numpy as np
import sys
import pickle
from pre_processing.main_preprocess import cluster, normalize, decay, createMultiview, saveFig

if __name__=="__main__":
    classes = open('dl_model/glosses','r',encoding='utf-8-sig').readlines()
    classes = [ gloss.strip() for gloss in classes ]

    _model= sys.argv[1]

    torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    # np.random.seed(1)
    torch.backends.cudnn.deterministic = True

    net=wordNet(2048,len(classes),2,5,0.65,False,10,True)
    modelPath='./dl_model/checkpoints/'+_model
    net.load_state_dict(torch.load(modelPath,map_location='cpu'),strict=False)
    m=nn.Softmax(dim=1)
    predictions=[]
    labels=[]
    net.eval()
    
    # filename='./data/outdoor/alma_eye_eye_different_01.pkl'
    filename='./data/outdoor/alma_feel_lazy_14.pkl'

    with open(filename, "rb") as pm_data:
        pm_contents = pickle.load(pm_data, encoding="bytes")

    f = 10 #number of frames

    #Outlier Removal and Translation
    c = 0
    for key, pts in pm_contents.items():
        c += len(pts)
        pm_contents[key] = cluster(pts, e = 0.8, outlier=True)
        pm_contents[key] = normalize(pts)

    # Aggregate Frames
    aggframes = decay(pm_contents, c, f)


    # Cluster
    clustFrames = []                        # array to contain dictionaries
    for _, xyz in aggframes.items():        # iterated len(aggframes) times which is num of frames
        if(c < 10):
            clustFrames.append(dict({0:xyz}))
            continue
        clust = cluster(xyz, e = 0.5, min_samp = 5, outlier=True)       # dictionary with key color c,and item of np array size n x 3 (pts)
        if(len(clust) == 0): continue
        clustFrames.append(clust)

    while(len(clustFrames) < 10):
        clustFrames.append(dict({0:np.zeros((1,3))}))

    # Multiview
    data = {'xy': [], 'yz': [], 'xz': []} 
    for _3dframe in clustFrames:
        xyf,yzf,xzf = createMultiview(_3dframe)      # tuple containing 3 dictionaries xy, yz, xz 2dframes
        if(len(xyf.values()) == 0): continue
        data['xy'].append(np.concatenate(list(xyf.values())))
        data['yz'].append(np.concatenate(list(yzf.values())))
        data['xz'].append(np.concatenate(list(xzf.values())))

    outData = {'xy': [], 'yz': [], 'xz': []}
    for view in data.keys():
        for frame in data[view]:
            outData[view].append(saveFig(frame, axis=view, reSize=True, saveNumpy=True))
        
    data = [np.concatenate((outData['xy'], outData['yz'], outData['xz']), axis=0)]
    # print(data)
    preprocessed = torch.tensor(data, dtype=torch.float32)

    # model predict
    o = net(preprocessed, 10)
    prediction = torch.max(m(o), dim=1)[1].cpu().numpy().tolist()
    print(prediction)