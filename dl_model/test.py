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

if __name__=="__main__":
# """
#     run the file: test.py datasetfolder modelpath

#     e.g.
#     Enter in cmd line:
#         python test.py outdoor outdoor-to-outdoor/latest-model.pth
#     to signify:
#         - use the filePath: outdoor_train_test_all_glosses
#         - use modelpath: outdoor-to-outdoor/latest-model.pth

# """

    setup, _model= sys.argv[1], sys.argv[2]

    classes = open('glosses','r',encoding='utf-8-sig').readlines()
    classes = [ gloss.strip() for gloss in classes ]
    # _10classes = ['why', 'help_you', 'important', 'family', 'improve', 'none', 'batangas', 'corruption', 'body', 'graduate']

    # filePath = setup + '_train_test_all_glosses'
    # dirPath = '../data/preprocessed_data/' + setup

    filePath = '_'.join([setup,'train_test_all_glosses'])
    dirPath = '../data/preprocessed_data/'
    
    dirPath = dirPath if(setup == 'combined') else dirPath + setup
    
    print(dirPath, filePath)

    testDataset=multiViewDatasetConcat(dirPath,classes,filePath,combined=(setup=='combined'),train=False,frameCount=10,wordOnly=True)

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    torch.backends.cudnn.deterministic = True

    net=wordNet(2048,len(classes),2,5,0.65,False,10,True)
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
        o=net(x,10)	
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
    figFile = 'test-%s_model-%s'%(setup,_model.split('/')[0])
    title = "Train: %s; Test: %s; Accuracy: %f"%(_model.split('-')[0], setup, accuracy)
    plt.title(title)
    plt.savefig(os.path.join('confusion_matrices',figFile + '.png'))