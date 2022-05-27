import os
import pandas as pd
import numpy as np
from rsa import sign
import torch
import multiprocessing as mp
import glob
from torch.utils.data import Dataset
from string import digits
import logging 
from sklearn.metrics import accuracy_score,confusion_matrix

class multiViewDataset(Dataset):
	def __init__(self, dirPath, classes, filePath, train=True, frameCount=40):
		self.dirPath = dirPath
		self.classes = classes
		self.fileList = []
		self.trainOnly = train
		self.data = None
		self.labels = []
		self.frameCount = frameCount
		self.views = ['xy','yz','xz']

		f = open(filePath,'r')
		f = f.readlines()
		if train:
			f=[f.strip().split(',')[1] for f in f if 'Train' in f]
		else:
			f=[f.strip().split(',')[1] for f in f if 'Test' in f]
		
		f = [getUser(f) + '_' + getLabel(f) for f in f]
		self.fileList = f
		self.fileListLow = [f.lower() for f in f]
		
		inDirs = glob.glob(dirPath+'/*')
		inDirs = [inDir for inDir in inDirs if inDir.split('/')[-1] in self.fileList or inDir.split('/')[-1] in self.fileListLow]
		self.data, self.labels = self.loadData(inDirs)

	def __len__(self):
		return len(self.labels)
		
	def __getitem__(self,idx):
		return torch.tensor(self.data['xy'][idx],dtype=torch.float32),torch.tensor(self.data['yz'][idx],dtype=torch.float32),torch.tensor(self.data['xz'][idx],dtype=torch.float32),torch.tensor(self.labels[idx],dtype=torch.long)
	
	def loadData(self,inDirs):
		data={'xy':[],'yz':[],'xz':[]}
		labels=[]

		for inDir in inDirs:
			npFiles=glob.glob(inDir+'/*')
			npFiles=sorted(npFiles)
			data_set=False
			for i in range(0,len(npFiles),3):
				view=npFiles[i].split('/')[-1].split('-')[0]
				body=np.load(npFiles[i])
				left=np.load(npFiles[i+1])
				right=np.load(npFiles[i+2])
				if body.shape[0] < self.frameCount:
					if self.logger is not None:
						self.logger.info("For files %s the frame count is %s",npFile,body.shape[0])
					continue
				elif body.shape[0] > self.frameCount:
					start=int(2*(body.shape[0]-self.frameCount)/3)
				else:
					start=0
				data[view].append(np.concatenate((body[start:start+self.frameCount],left[start:start+self.frameCount],right[start:start+self.frameCount]),axis=0))
				data_set=True
			if data_set:
				labels.append(self.classes.index(getLabel(inDir,self.classes)))
			
		# data -> dictionary 
		# labels -> array
		return data,labels

def getUser(file):
	user = file.strip().split('/')[-1].split('_')[0]
	return user

def getLabel(file, classes=None):
	pass

# class SignsDataset(torch.utils.data.Dataset):
#     def __init__(self, root, classes, csv_file):
#         self.root = root 

#         # Initialize data
#         self.df = pd.read_csv(csv_file)
        
#         df_values = self.df.values

#         # df: ['time_series file', 'class_id']
#         self.signs = np.unique(np.unique(self.df.iloc[:, 0].values))
#         self.sign_gloss_dict = {}

#         # for every time_series_file load into an array
#         for data in df_values:
#             fname = data[0]
#             sign =  np.array([1,2,3,4,5])       # change to time_series_file loaded into an array
            
#             gloss =  data[1]                    # label
            
#             self.sign_gloss_dict[fname] = {
#                 'sign': sign,
#                 'gloss': gloss
#             }


        
#     def __len__(self):
#         # return the size of the dataset
#         return len(self.df)

#     def __getitem__(self, idx):
#         # fetching a data sample for a given key
#         return self.sign_gloss_dict[idx]