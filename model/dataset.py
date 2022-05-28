import os
import pandas as pd
import numpy as np
import torch
import multiprocessing as mp
import glob
from torch.utils.data import Dataset
from string import digits
import logging 
from PIL import Image
# from sklearn.metrics import accuracy_score,confusion_matrix

class multiViewDataset(Dataset):
	def __init__(self, dirPath, users, classes, train=True, frameCount=40, device=torch.device('cuda')):
		self.dirPath = dirPath
		self.users = users
		self.classes = classes
		self.fileList = []
		self.train = train
		self.data = None
		self.labels = []
		self.frameCount = frameCount
		self.device=device
		self.views = ['xy','yz','xz']
		if self.train:
			self.sample_range = range(1, 13)
		else:
			self.sample_range = range(13, 16)

		self.data={'xy': [], 'yz': [], 'xz': []}
		self.labels = []
		
		count=0
		for user in self.users:
			for label in self.classes:
				for i in self.sample_range:
					for view in self.views:
						frames = np.array([np.array(Image.open(self.dirPath+'/'+user+'/'+label+'/'+str(i)+'/'+view+'/'+str(frame)+'.png')) for frame in range(self.frameCount)])
						self.data[view].append(frames)
					self.labels.append(self.classes.index(label))

	def __len__(self):
		return len(self.labels)
		
	def __getitem__(self,idx):
		return torch.tensor(self.data['xy'][idx], dtype=torch.float32), torch.tensor(self.data['yz'][idx], dtype=torch.float32), torch.tensor(self.data['xz'][idx],dtype=torch.float32), torch.tensor(self.labels[idx],dtype=torch.long, device=self.device)
	

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