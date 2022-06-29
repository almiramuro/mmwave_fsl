import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pickle
import os
import sys
from matplotlib.ticker import NullLocator
from PIL import Image
import io

# from lib.plot import *s
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

    epts = extractedPts[:,:3]
    
    if(len(extractedPts[0]) == 4):
        sw = extractedPts[:,3]
    else:
        sw = None
    db = DBSCAN(eps=e, min_samples=min_samp).fit(epts)
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

        xyz = epts[class_member_mask & core_samples_mask]
        
        xyz = np.vstack((xyz,epts[class_member_mask & ~core_samples_mask]))
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
    agg = {frame: np.zeros((0,3)) for frame in fcount}
    
    start = 0
    frame = 1
    pts_perframe = int(k/f) if(k/f > 1) else 1
    
    # print(fcount,pts_perframe)
    # print(agg)

    for key, pts in raw.items():  
        for pt in pts:
            agg[frame] = np.vstack((agg[frame], pt))
            if(len(agg[frame]) == pts_perframe and frame < f): 
                frame += 1
            elif(len(agg[frame]) == pts_perframe and frame == f):
                return agg
        # print('numframes: %d'%frame)
        # print('\nupdated: ',agg)
    
    # check if all arrays have laman
    for i in fcount:
        if(agg[i].shape[0] == 0):
            agg[i] = np.vstack((agg[i], np.zeros((1,3))))
    print("agg items: ",len(agg.items()))
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

    return xyframes,yzframes,xzframes

def saveFig(x, axis = None, pltTitle = None, saveDir = '.', reSize = True, saveNumpy = False):
	# saveFigure(x,pltTitle='Image-'+str(count),saveDir=saveDir+'/'+axis, axis=axis,saveNumpy=saveNumpy,reSize=True)
	
	#Change axis limits
	fig=plt.figure()
	ax=fig.add_subplot(111)
	if(axis=='xy' or axis == None):
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_xlim(-0.5, 0.8)
		ax.set_ylim(-1.2, 0.6)
	elif(axis == 'yz'):
		ax.set_xlabel('y')
		ax.set_ylabel('z')
		ax.set_xlim(-1.2, 0.6)
		ax.set_ylim(-1.4, 1.1)
	elif(axis == 'xz'):
		ax.set_xlabel('x')
		ax.set_ylabel('z')
		ax.set_xlim(-0.5, 0.8)
		ax.set_ylim(-1.4, 1.1)

	if x is None:
		plt.clf()
	else:
		ax.scatter(x[:,0],x[:,1],c='red',marker='o')

	ax.set_axis_off()
	plt.margins(0,0)
	ax.xaxis.set_major_locator(NullLocator())
	ax.yaxis.set_major_locator(NullLocator())
	plt.ioff()
	plt.axis('off')
	if reSize:
		ram=io.BytesIO()
		plt.savefig(ram,format='jpeg')
		im=Image.open(ram)
		# im=im.convert('1')
		im=im.resize((100,74),Image.LANCZOS)	
		if saveNumpy:
			ram.close()
			plt.close()
			return np.asarray(im)
		# print(saveDir)
		# exit()
		im.save(os.path.join(saveDir,pltTitle+'.png'))
		ram.close()
		plt.close()
	else:
		plt.savefig(os.path.join(saveDir,pltTitle+'.jpeg'))
		plt.close()

def npySave(view, data, dataSaveDir=None, imgSaveDir=None):
    """
        view = str, any one of ['xy','yz','xz']
        data = all aggregated frames
        dataSaveDir = path (os.path.join(currDir, subfolder, place, filename[:-4]))
                        (sample: ./data/preprocessed_data/indoor_10_signs_15_reps/aaron_batangas_1)
        imgSaveDir = path for image
    """
    outData = []
    count = 1

    print('data has length of: %d'%len(data))
    for frame in data:
        if(imgSaveDir != None):
            imgSaveDir.replace('\\','/')
            viewDir = os.path.join(imgSaveDir,view)
            os.makedirs(viewDir, exist_ok=True)
            saveFig(frame,axis=view,pltTitle='Frame-'+str(count), saveDir=viewDir, reSize = True)
        outData.append(saveFig(frame, axis=view, reSize = True, saveNumpy = True))
        count += 1
    if(dataSaveDir != None):
        np.save(dataSaveDir+'/'+view+'.npy', np.array(outData))
    # print(outData)
    return np.array(outData)

def createTrainTestFile(dataFolder, raw_data, dataRatio):
    modelFolder = os.path.join('..','dl_model')

    outFileTxt = dataFolder+'_train_test_all_glosses'
    
    towrite = []

    # print(dataRatio[0])

    count = 0
    for d in range(len(raw_data)):
        if(count == dataRatio[0]+dataRatio[1]): count = 0
        if(count < dataRatio[0]):
            towrite.append("Train,"+raw_data[d])
        else:
            towrite.append("Test,"+raw_data[d])
        count += 1

    out = open(os.path.join(modelFolder,outFileTxt), 'w')
    out.writelines('\n'.join(towrite))    

    # print(os.listdir(modelFolder))

def preprocess(processDir, filename, f, saveData=True, saveImg=False): 

    # Input handling
    """
        processDir = raw files to process directory 
        filename = string: <name>_<gloss>_<num>.pkl
        f = number of desired aggregated frames
    """
    
    with open(os.path.join(processDir,filename),"rb") as pm_data:
        pm_contents = pickle.load(pm_data,encoding ="bytes")

    # Outlier Removal and Translation
    N = 0
    for key, pts in pm_contents.items():
        N += len(pts)
        pm_contents[key] = cluster(pts, e = 0.8, outlier=True)
        # pm_contents[key] = normalize(pts)
    newN = len(pm_contents.items())
    print('orig # of pts: %d || new # of pts: %d'%(N,newN))

    # Aggregate Frames
    aggframes = decay(pm_contents, N, f)

    # Cluster

    clustFrames = []                        # array to contain dictionaries
    for _, xyz in aggframes.items():        # iterated len(aggframes) times which is num of frames
        if(newN <= f):
            clustFrames.append(dict({0:xyz}))
            continue
        # clust = cluster(xyz, e = 0.5, min_samp = 5, outlier=True)       # dictionary with key color c,and item of np array size n x 3 (pts)
        # print(clust)
        if(len(clust) == 0): continue
        clustFrames.append(clust)

    while(len(clustFrames) < f):
        clustFrames.append(dict({0:np.zeros((1,3))}))

    if(len(clustFrames) != f): 
        print('frames not %d but %d'%(f,len(clustFrames)))
        sys.exit()
    # print(clustFrames)

    # Output handling 
    """
        numpy file (.npy)
        1 npy contains 1 array with size (# of frames, # of pts/frame, # of dimensions/pt)
        # of frames = f
        # of pts/frame = dapat constant
        # of dimensions/pt = 2             (x,y)
    """
    data = {'xy': [], 'yz': [], 'xz': []} 
    for _3dframe in clustFrames:
        # print("------------>3D FRAME: ",_3dframe)
        xyf,yzf,xzf = createMultiview(_3dframe)      # tuple containing 3 dictionaries xy, yz, xz 2dframes
        # print('len per framee: %d'%len(_3dframe.items()))
        if(len(xyf.values()) == 0): continue
        data['xy'].append(np.concatenate(list(xyf.values())))
        data['yz'].append(np.concatenate(list(yzf.values())))
        data['xz'].append(np.concatenate(list(xzf.values())))

    # print('data: ', data)
    # return
    dataDir = processDir.split('/')[1]                          #dirPath (data)
    
    outPath = 'preprocessed_data'    
    imgPath = 'images'

    outFolder = processDir.split('/')[2]
    imgFolder = processDir.split('/')[2]

    # print(filename[:-4])

    dataSaveDir = os.path.join('..',dataDir, outPath, outFolder , filename[:-4])            #(data/preprocessed_data/place/user_gloss_it)
    imgSaveDir = os.path.join('..',dataDir, imgPath, imgFolder, filename[:-4])              #(data/images/place/user_gloss_it)

    # for data
    os.makedirs(dataSaveDir, exist_ok=True)                 #inDirs (data/preprocessed_data/place)
                                                            #inDir (data/preprocessed_data/place/user_gloss_it)
    
    # for imgs    
    os.makedirs(imgSaveDir, exist_ok=True)                   #inDirs (data/preprocessed_data/place)
                                                             #inDir (data/preprocessed_data/place/user_gloss_it)
                             
    views = ['xy', 'yz', 'xz']

    img = None if(saveImg == False) else imgSaveDir
    npdata = None if(saveData == False) else dataSaveDir
    
    for view in views:
        npySave(view, data[view], npdata, img)
    
    
if __name__=="__main__":
    """
        run the file: main_preprocess.py rawfilepath_ ratio 

        e.g.
        Enter in cmd line:
            python main_preprocess.py outdoor_24_signs_15_reps 80 20 
        to signify: 
            - the folder to process is outdoor_24_signs_15_reps 
            - use a 80:20 ratio for train and test files
    """
    saveData = True
    saveImg = input("Enter Y to save preprocessed images: ")
    saveImg = True if(saveImg.upper()=="Y") else False

    dataDir = '../data/'
    dataFolder = sys.argv[1]
    ratio = (float(sys.argv[2])/100, float(sys.argv[3])/100)
    repetitions = 15
    dataRatio = tuple(int(repetitions*r) for r in ratio) 
    
    processDir = os.path.join(dataDir,dataFolder)
    raw_data = os.listdir(processDir)

    print('processDir: %s'%(processDir))

    # print(len(raw_data))
    processed_data = []
    for file in raw_data:
        if(file[-4:] != '.pkl'): continue
        if(saveData == True):
            pathCheck = os.path.join(dataDir.replace('/','\\'),'preprocessed_data',dataFolder,file[:-4])
            if(os.path.isdir(pathCheck)): continue

        print('preprocessing %s with contents: '%file)
        preprocess(processDir,file, 5, saveData, saveImg)
        processed_data.append(file)
        
    
    # createTrainTestFile(dataFolder,raw_data,dataRatio)