from utility import createTrainTest,multiViewDataset,createLogger
from torch.utils.data import DataLoader
from model import lowResNet
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch

userDirs=['../arisa_23words','../ari_23words','../jesse_23words','../riley_new23words']
users=['riley','ari','arisa','jesse']
testCount=2
outFile='train_test_all'
classes=['teach','you','me','piano','want','very','angry','fire','everyone','huddle','how','today',
	'weather','wakeup','grandmother','never','there','actually','have','must','worried','they','visiting','students']
#createTrainTest(userDirs,users,testCount,outFile,classes)
filePath='/mnt/d/impactProject/nn_models/train_test_all'
dirPath='/mnt/d/impactProject/ali_23words_out'
trainDataset=multiViewDataset(dirPath,classes,filePath)
saveDir='/scratch/psanthal/'
'''
lowResnet Parameters
hidden_dim,class_size,num_layers,batch_size,dropout,use_cuda,frameCount
'''
net=lowResNet(2048,len(classes),2,10,0,False,40)
optimizer=optim.Adam(net.parameters(),lr=0.000001)
multiViewDataLoader=DataLoader(trainDataset,10,shuffle=True)
criterion=nn.CrossEntropyLoss()
logger=createLogger('./','temp-log')
for epoch in range(100):
	running_loss=0
	batchCount=0
	for xy,yz,xz,label in multiViewDataLoader:		
		y=net({'xy':xy,'yz':yz,'xz':xz})
		loss=criterion(y,label)
		loss.backward()
		optimizer.step()
		running_loss+=loss.item()
		batchCount+=1
		if batchCount==1:
			logger.info("Loss for epoch:%s is: %s",epoch,(running_loss/batchCount))
			batchCount=0
			running_loss=0
	if epoch%10==0 and epoch > 0:
		torch.save(net.state_dict(),saveDir+'model-'+str(epoch)+'.pth')

