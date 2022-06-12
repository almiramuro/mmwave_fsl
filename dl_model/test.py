from utility import createLogger,multiViewDatasetConcat,computeAccuracy
from torch.utils.data import DataLoader
from model import wordNet
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F

users = ['aaron', 'mira', 'luis']

classes = ['why', 'help_you', 'important', 'family', 'improve', 'none', 'batangas', 'corruption', 'body', 'graduate']

filePath='train_test_all_glosses'
dirPath = '../data/preprocessed_data/'

testDataset=multiViewDatasetConcat(dirPath,classes,filePath,train=False,frameCount=10,wordOnly=True)

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True

net=wordNet(2048,len(classes),2,5,0.65,False,10,True)
modelPath='./latest-model.pth'
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
print("The accuracy for unclustered word  model with dropout and highRes: %f is: %f"%(390,accuracy))
print("The confusion Matrix is")
print(confusion)