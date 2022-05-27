import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from PIL import Image
import io
import os
import pandas as pd

def saveFig(x, axis = None, pltTitle = 'temp', saveDir = '.', reSize = True, saveNumpy = False):
       
    dir = pltTitle.split('_')
    while len(dir) > 5:
        dir[1] = '_'.join(dir[1:3:])
        del dir[2]
    
    pltTitle = dir[-1]
    
    dir = '/'.join(dir[:-1:])
    saveDir = os.path.join(saveDir, dir)
    if not os.path.isdir(saveDir):
        os.makedirs(saveDir)
    
    #Change axis limits
    fig=plt.figure()
    ax=fig.add_subplot(111)
    if axis=='xy' or axis == None:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-0.477930, -0.003037)
        ax.set_ylim(-1.145667, 0.036919)
    elif axis == 'yz':
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        ax.set_xlim(-1.145667, 0.036919)
        ax.set_ylim(-1.354615, 0.002628)
    elif axis == 'xz':
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_xlim(-0.477930, -0.003037)
        ax.set_ylim(-1.354615, 0.002628)
    if x is None:
        plt.clf()
    else:
        ax.scatter(x[:,0],x[:,1],c='black',marker='o')
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
        im=im.convert('1')
        im=im.resize((40,29),Image.LANCZOS)	
        if saveNumpy:
            ram.close()
            plt.close()
            return np.asarray(im)
        # print(saveDir)
        # exit()
        im.save(saveDir+'/'+pltTitle+'.png')
        ram.close()
        plt.close()
    else:
        plt.savefig(saveDir+'/'+pltTitle+'.jpeg')
        plt.close()

def toFloat(pair = [None, None]):
    out = []
    if len(pair) > 2:
        count = len(pair) - 2
        while count:
            if '\r' in pair:
                pair.remove('\r')    
            else:
                pair.remove('')
            count -= 1
    for i in [0,1]:
        if 'e' in pair[i]:
            a = pair[i].split('e')
            if a[-1][0] == '+':
                out.append(float(a[0])*(10^(int(a[-1][1::]))))
            elif a[-1][0] == '-':
                out.append(float(a[0])/(10**(int(a[-1][1::]))))
        else:
            out.append(float(pair[i]))        
    out = np.array(out, dtype=float)
    return out
            
def npArray(data, view = None):
    out = []
    for i in range (0, data.size):
        frame = []
        for j in range (0, len(data[i].split(' \n '))):
            array = np.array(data[i].split(' \n ')[j].split('\n '))
            array2d = []
            for k in range (0, array.size):
                array[k] = array[k].replace('[', '')
                array[k] = array[k].replace(']', '')
                # print(array[k].split(' '))
                array2d.append(toFloat(array[k].split(' ')))
            # exit()                    
            frame.append(np.array(array2d))
        out.append(np.array(frame, dtype=np.ndarray))
    # print(out[0].size)
    # out = np.array(out, dtype=np.ndarray)
    # print(out)
    return out

def main():
    filePath = 'data/preprocessed_data'
    saveDir = 'data/images'
    if not os.path.isdir(saveDir):
        os.makedirs(saveDir)
    signers = ['aaron', 'mira', 'luis']
    signs = ['why', 'help_you', 'important', 'family', 'improve', 'none', 'batangas', 'corruption', 'body', 'graduate']
    dataset = []
    for signer in signers:
        for sign in signs:
            sample = 15
            while sample:
                dataset.append('{}_{}_{}_processed.csv'.format(signer, sign, sample))
                sample -= 1
    for data in dataset:
        file = filePath + '/' + data
        d = pd.read_csv(file)
        views = {'xy': [], 'yz': [], 'xz': []}
        views['xy'] = (npArray(d.xy.values, 'xy'))
        views['yz'] = (npArray(d.yz.values, 'yz'))
        views['xz'] = (npArray(d.xz.values, 'xz'))
        for view in views:
            for frame in range (0, len(views[view])):
                # print(len(views[view][frame][0]))
                saveFig(views[view][frame][0], axis=view, pltTitle = data[:-13:]+view+'_'+str(frame), saveDir = saveDir, reSize = True, saveNumpy = False)
        # exit()
    # file = 'aaron_hello_world_2_processed.csv'
    # d = pd.read_csv(file)
    # views = {'xy': [], 'yz': [], 'xz': []}
    # views['xy'] = npArray(d.xy.values, 'xy')
    # views['yz'] = npArray(d.yz.values, 'yz')
    # views['xz'] = npArray(d.xz.values, 'xz')
    # for view in views:
    #     for frame in range (0, len(views[view])):
    #         saveFig(views[view][frame], axis=view, pltTitle = file[:-13:]+view, saveDir = saveDir, reSize = True, saveNumpy = False)
            
if __name__ == '__main__':
    main()