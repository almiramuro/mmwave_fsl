import enum
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import multiViewDataset
from model import Net

dirPath = '../data/images'
users = ['aaron', 'mira', 'luis']
classes = ['why', 'help_you', 'important', 'family', 'improve', 'none', 'Batangas', 'corruption', 'body', 'graduate']
saveDir='./checkpoints/'

def train():
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	print('Device selected: ', device)

	# Define datasets
	'''
	multiViewDataset Parameters
	dirPath, classes, filePath, train=True, frameCount=40
	'''
	trainDataset = multiViewDataset(dirPath, users, classes, train=True, frameCount=20, device=device)
	print('Train dataset loaded')
	testDataset = multiViewDataset(dirPath, users, classes, train=False, frameCount=20, device=device)
	print('Test dataset loaded')

	# Define dataloaders
	trainDataLoader = DataLoader(trainDataset, batch_size=10, shuffle=True)
	# testDataLoader = DataLoader(testDataset, batch_size=1, shuffle=False)

	# Define model
	'''
	model parameters
	class_size, hidden_dim=2048, num_layers=2, dropout=0.65
	'''
	model = Net(len(classes),2048, 2, 0.65, frameCount=20, device=device)
	model.to(device)
	print('Model loaded')

	# # Define optimizer
	optimizer = optim.Adam(model.parameters(), lr=0.000001)
	criterion = nn.CrossEntropyLoss().to(device)

	# Train loop
	for epoch in range(100):
		running_loss = 0
		batchCount = 0
		for xy, yz, xz, label in trainDataLoader:	
			# print(xy.shape, label)
			# exit()
			y = model({'xy': xy, 'yz': yz, 'xz': xz})
			# print(y.is_cuda)
			# print(label.is_cuda)
			# exit()
			loss = criterion(y, label)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			batchCount += 1
			if batchCount == 1:
				print("Loss for epoch:{} is: {}".format(epoch,(running_loss/batchCount)))
				batchCount =0
				running_loss = 0
		if epoch % 10 == 0 and epoch > 0:
			torch.save(model.state_dict(), saveDir+'model-'+str(epoch)+'.pth')
		
		#eval

if __name__=='__main__':
	train()
