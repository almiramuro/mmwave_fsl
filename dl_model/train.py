import os
import sys
from utility import createTrainTest,localize,multiViewDatasetConcat,computeAccuracy
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
if(sys.platform == 'linux'):
	from model import wordNet
else:
	from dl_model.model import wordNet
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F


if __name__=="__main__":
	"""
        run the file: train.py datasetfolder

        e.g.
        Enter in cmd line:
            python train.py outdoor_24 train-luishome test-floor2 
			
		to use the filePath outdoor_24_train_test_all_glosses for training
		to use the filePath outdoor_24_test_all_glosses for testing

    """
	# Text files 
	trainFilePath = '_'.join([sys.argv[1],sys.argv[2].split('-')[1],'train_test'])		#outdoor_24_luishome
	testFilePath = '_'.join([sys.argv[1],sys.argv[3].split('-')[1], 'train_test'])		#outdoor_24_luishome
		
	dirPath = '../data/'

	# users = ['aaron', 'mira', 'luis']
	
	_24classes = ['hello', 'nice_meet_you', 'good_night', 'sign_name', 'how_you', 'why', 'sleep', 'calendar', 'cook', 'computer', 'help_you', 'important', 'family', 'improve', 'none', 'batangas', 'bulacan', 'bicol', 'flood', 'go_home', 'corruption', 'body', 'life', 'graduate']
	_10classes = ['why', 'help_you', 'important', 'family', 'improve', 'none', 'batangas', 'corruption', 'body', 'graduate']
	saveDir='./checkpoints/'
	os.makedirs(saveDir, exist_ok=True)

	classes = _10classes if(trainFilePath.split('_')[1] == '10') else _24classes
	
	trainDirPath = '../data/preprocessed_data/'+ trainFilePath.removesuffix('_train_test')
	testDirPath = '../data/preprocessed_data/'+ testFilePath.removesuffix('_train_test')

	trainDataset=multiViewDatasetConcat(trainDirPath,classes,trainFilePath,train=True,frameCount=10,wordOnly=True)
	testDataset=multiViewDatasetConcat(testDirPath,classes,testFilePath,train=False,frameCount=10,wordOnly=True)

	torch.manual_seed(1)
	torch.cuda.manual_seed(1)
	np.random.seed(1)
	torch.backends.cudnn.deterministic = True


	net=wordNet(2048,len(classes),2,5,0.65,use_cuda=False,frameCount=10,dataParallel=True)		# 40 changed to 10 ; 10 changed to 5

	optimizer=optim.Adam(net.parameters(),lr=0.000001)
	trainDataLoader=DataLoader(trainDataset, batch_size=5, shuffle=True)
	testDataLoader=DataLoader(testDataset,5,shuffle=False)
	criterion=nn.CrossEntropyLoss()
	net.train()
	leadingAccuracy = 0
	for epoch in range(400):
		running_loss=0
		batchCount=0
		for x,label in trainDataLoader:		
			net.zero_grad()
			y=net(localize(x),10)		# 40 changed to 10
			loss=criterion(y,localize(label))
			loss.backward()
			optimizer.step()
			running_loss+=loss.item()
			batchCount+=1
			if batchCount==5:
				print("Loss for epoch:%d is: %f"%(epoch,running_loss/(batchCount*5)))
				batchCount=0
				running_loss=0

		if(epoch%5==0 and epoch > 0):
			torch.save(net.state_dict(),saveDir+'model-'+str(epoch)+'.pth')
			
			_newmodel = 'model-'+str(epoch)+'.pth'
			m=nn.Softmax(dim=1)
			predictions=[]
			labels=[]
			net.eval()

			count = 0
			for x,label in testDataLoader:
				print('sample ', count)		
				o=net(localize(x),10)	
				prediction = torch.max(m(o),dim=1)[1].cpu().numpy().tolist()
				predictions += prediction
				labels+=label.cpu().numpy().tolist()
				print(prediction, label)
				count += 1

			confusion,accuracy=computeAccuracy(labels,predictions,[i for i in range(len(classes))])
			print("The accuracy for %s using %s is: %f"%(testFilePath,_newmodel,accuracy))
			
			if(leadingAccuracy == 0):
				leadingModel = _newmodel
				continue
			
			if(accuracy > leadingAccuracy):
				os.remove(leadingModel)
				leadingModel = _newmodel
				leadingAccuracy = accuracy
			elif(accuracy < leadingAccuracy):
				os.remove(_newmodel)