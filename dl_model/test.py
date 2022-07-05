from utility import multiViewDatasetConcat,computeAccuracy
from torch.utils.data import DataLoader
from model import wordNet
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F
import sys
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
# from pre_processing.main_preprocess import frameCount as frames 

frames = 10

if __name__=="__main__":
    """
        run the file: test.py datasetfolder modelpath

        e.g.
        Enter in cmd line:
            python test.py outdoor outdoor-to-outdoor/latest-model.pth
        to signify:
            - use the filePath: outdoor_train_test_all_glosses
            - use modelpath: outdoor-to-outdoor/latest-model.pth

    """

    setup, _model= sys.argv[1], sys.argv[2]
    signer = [('alma','A'),('melissa','C'),('rommel','B'), (None,'All')][3]

    classes = open('glosses','r',encoding='utf-8-sig').readlines()
    classes = [ gloss.strip() for gloss in classes ]
    # _10classes = ['why', 'help_you', 'important', 'family', 'improve', 'none', 'batangas', 'corruption', 'body', 'graduate']
    
    filePath = '_'.join([setup,'train_test_all_glosses'])
    dirPath = '../data/preprocessed_data/'
    
    dirPath = dirPath if(setup == 'combined') else dirPath + setup
    comb = True if(setup=='combined') else False
    print(dirPath, filePath)
    testDataset=multiViewDatasetConcat(dirPath,classes,filePath,combined=comb,train=False,frameCount=frames,wordOnly=True,signer=signer[0])

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    torch.backends.cudnn.deterministic = True

    net=wordNet(2048,len(classes),2,5,0.65,False,frames,True)
    modelPath='./checkpoints/'+_model
    net.load_state_dict(torch.load(modelPath,map_location='cpu'),strict=False)
    m=nn.Softmax(dim=1)
    predictions=[]
    labels=[]
    net.eval()
    multiViewDataLoader=DataLoader(testDataset,5,shuffle=False)
    count = 0
    for x,label in multiViewDataLoader:
        print('sample ', count)		
        o=net(x,frames)	
        prediction = torch.max(m(o),dim=1)[1].cpu().numpy().tolist()
        predictions += prediction
        labels+=label.cpu().numpy().tolist()
        print(prediction, label)
        count += 1
    confusion,accuracy=computeAccuracy(labels,predictions,[i for i in range(len(classes))])
    print("The accuracy for %s using %s is: %f"%(setup,_model,accuracy))
    print("The Confusion Matrix is")
    print(confusion)
    ConfusionMatrixDisplay.from_predictions(labels,predictions)
    figFile = 'signer-%s-test-%s_model-%s'%(signer[1],setup,_model.split('/')[0])
    title = "Signer: %s\n Train: %s; Test: %s; Accuracy: %f"%(signer[1],_model.split('-')[0], setup, accuracy)
    plt.title(title)
    plt.savefig(os.path.join('confusion_matrices',figFile + '.png'))