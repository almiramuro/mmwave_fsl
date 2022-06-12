from utility import createTrainTest,createLogger,multiViewDatasetConcat
from torch.utils.data import DataLoader
from dl_model.model import wordNet
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F
import os
import sys

if __name__=="__main__":
	"""
        run the file: train.py	datasetfolder

        e.g.
        Enter in cmd line:
            python train.py outdoor_24_signs_15_reps 
			
		to use the filePath outdoor_24_train_test_all_glosses

    """
	setup = sys.argv[1]
	
	dirPath = '../data/'
	users = ['aaron', 'mira', 'luis']
	classes = ['hello', 'nice_meet_you', 'good night', 'sign_name', 'how_you', 'why', 'sleep', 'calendar', 'cook', 'computer', 'help_you', 'important', 'family', 'improve', 'none', 'batangas', 'bulacan', 'bicol', 'flood', 'go_home', 'corruption', 'body', 'life', 'graduate']
	saveDir='./checkpoints/'
	os.makedirs(saveDir, exist_ok=True)


	filePath = '_'.join([setup.split('_')[i] for i in range(2)]) + '_train_test_all_glosses'

	dirPath = '../data/preprocessed_data/'

	trainDataset=multiViewDatasetConcat(dirPath,classes,filePath,train=True,frameCount=10,wordOnly=True)

	# logger=createLogger('./scratch','unclustered-gloss-all')

	# logger.info("Training set total number of samples:%s",len(trainDataset))

	torch.manual_seed(1)
	torch.cuda.manual_seed(1)
	np.random.seed(1)
	torch.backends.cudnn.deterministic = True


	net=wordNet(2048,len(classes),2,5,0.65,False,10,True)		# 40 changed to 10 ; 10 changed to 5

	optimizer=optim.Adam(net.parameters(),lr=0.000001)
	multiViewDataLoader=DataLoader(trainDataset, batch_size=5, shuffle=True)
	criterion=nn.CrossEntropyLoss()
	net.train()
	for epoch in range(400):
		running_loss=0
		batchCount=0
		for x,label in multiViewDataLoader:		
			net.zero_grad()
			y=net(x.cuda(),10)		# 40 changed to 10
			loss=criterion(y,label.cuda())
			loss.backward()
			optimizer.step()
			running_loss+=loss.item()
			batchCount+=1
			if batchCount==5:
				print("Loss for epoch:%d is: %d",epoch,(running_loss/(batchCount*5)))
				batchCount=0
				running_loss=0
		if epoch%5==0 and epoch > 0:
			torch.save(net.state_dict(),saveDir+'model-'+str(epoch)+'.pth')