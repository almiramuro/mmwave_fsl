import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from model.utility import multiViewDataset
from model.model import Net

userDirs=['../arisa_23words','../ari_23words','../jesse_23words','../riley_new23words']
users=['aaron', 'mira', 'luis']
testCount=2
outFile='train_test_all'
classes=['why', 'help_you', 'important', 'family', 'improve', 'none', 'Batangas', 'corruption', 'body', 'graduate']
#createTrainTest(userDirs,users,testCount,outFile,classes)
filePath='/mnt/d/impactProject/nn_models/train_test_all'
dirPath='/mnt/d/impactProject/ali_23words_out'
trainDataset=multiViewDataset(dirPath,classes,filePath)
saveDir='/scratch/psanthal/'




def train():
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	print('Device selected: ', device)

	# Define datasets
	'''
	multiViewDataset Parameters
	dirPath, classes, filePath, train=True, frameCount=40
	'''
	trainDataset = multiViewDataset(dirPath, classes, filePath, train=True, frameCount=40)
	testDataset = multiViewDataset(dirPath, classes, filePath, train=False, frameCount=40)

	# Define dataloaders
	trainDataLoader = DataLoader(trainDataset, batch_size=10, shuffle=True)
	testDataLoader = DataLoader(testDataset, batch_size=10, shuffle=False)

	# Define model
	'''
	model parameters
	class_size, hidden_dim=2048, num_layers=2, dropout=0.65
	'''
	model = Net(len(classes),2048, 2, 0.65)
	model.to(device)

	# Define optimizer
	optimizer = optim.Adam(model.parameters(), lr=0.000001)
	criterion = nn.CrossEntropyLoss()

	# Train loop
	for epoch in range(100):
		running_loss = 0
		batchCount = 0
		for xy, yz, xz, label in trainDataLoader:		
			y = model({'xy': xy, 'yz': yz, 'xz': xz})
			loss = criterion(y, label)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			batchCount += 1
			if batchCount == 1:
				print("Loss for epoch:%s is: %s",epoch,(running_loss/batchCount))
				batchCount =0
				running_loss = 0
		if epoch % 10 == 0 and epoch > 0:
			torch.save(model.state_dict(), saveDir+'model-'+str(epoch)+'.pth')
