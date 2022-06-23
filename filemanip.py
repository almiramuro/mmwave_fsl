import sys 
import os
import shutil
import pickle
import numpy as np

if __name__ == '__main__':
    
    dataDirs = ['indoor', 'indoor_wnoise', 'outdoor']
    processDirs = [os.path.join('data',dataDirs[i]) for i in range(3)]
    raw_files = [ os.listdir(processDirs[i]) for i in range(3)]

    # print(len(raw_files))
    # for i in range(len(raw_files)):
    #     print(len(raw_files[i]))

    src = []
    dst = []
    for i in range(3):
        for file in raw_files[i]:
            if('lazy' in file):
                # print(file)
                # os.rename(os.path.join(processDirs[i],file),os.path.join(processDirs[i], file.replace('real','reel')))
                src.append(os.path.join(processDirs[i],file))
                dst.append(os.path.join('feellazy',dataDirs[i],file))

    # # Separate the reel_lazy files
    # for i in range(len(src)):
    #     shutil.copy(src[i],dst[i])


    # See total points per pkl file
    numpoints = []

    for file in dst:
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
            n = 0
            for t, pts in data.items():
                n += len(pts)
        numpoints.append((file,n))          # .split('\\')[-1].split('_')[0]

    less10 = []
    for d in numpoints:
        if(d[1] < 10):
            print(d)
            less10.append(d[1])
    less10 = np.array(less10)
    print(len(less10))
    print(np.unique(less10))

        