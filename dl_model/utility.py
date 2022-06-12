﻿import multiprocessing as mp
import glob
from torch.utils.data import Dataset
from string import digits
import numpy as np
import torch
import logging 
from sklearn.metrics import accuracy_score,confusion_matrix
import itertools
from asr_evaluation import __main__
import os
import copy
from subprocess import PIPE, run
import torch.nn as nn
import torch
import torch.functional as F
from torchvision.utils import save_image
from PIL import Image

def doMultiProcessing(inFunc,inDir,split,arguments,noJobs=16):
	processes=[]
	count=0
	inFiles=glob.glob(inDir+'/*')
	for i in range(0,len(inFiles),split):
		p = mp.Process(target=inFunc,args=tuple([inFiles[0:0+split]]+arguments))
		if count > noJobs:
			for k in processes:
				k.join()
				processes = []
				count = 0
		p.start()
		count += 1
	if count > 0:
		for k in processes:
			k.join()
	return

def rmDigit(label):
	rm_dig=str.maketrans('','',digits)
	return label.translate(rm_dig)

def getLower(inString):
	return inString.lower()

def getLabel(inFile,nonManual=None,classes=None,withCount=False,includeMan=False,onlyNonManual=False,wordOnly=False):
	"""
		getLabel(inDir,classes=self.classes,wordOnly=True)
	"""
	if classes is not None:
		# This is what we care about
		if wordOnly:
			# This is what we care about
			try:
				return [k for k,v in enumerate(classes) if v in inFile][0] 
			except:
				return None
		
		if onlyNonManual:	# ignore for word only
			if any([label in nonManual for label in inFile.split('/')[-1].split('_')[1].split('-')[:-1]]):
				return nonManual.index(inFile.split('/')[-1].split('_')[1].split('-')[:-1][-1])
			else:
				return nonManual.index('manual')
		
		# segmenting the labels
		if 'teach' in inFile: 
			return [classes.index(label) for label in ['you','teachme']] 
		
		if any([label in nonManual for label in inFile.split('/')[-1].split('_')[1].split('-')[:-1]]):
			labels=inFile.split('/')[-1].split('_')[1].split('-')[:-2]
		else:
			labels=inFile.split('/')[-1].split('_')[1].split('-')[:-1]
		
		frmCls=None
		if "-i-"  in inFile:
			frmCls=labels.index('i')
		elif "-my-" in inFile:
			frmCls=labels.index('my')
		if frmCls is not None:
			labels[frmCls]='me'



		if any([label not in classes for label in labels]):
			# if any of the elements inside of the any function (a label in labels not in classes)
			return None
		else:
			# all label in labels are in classes
			return [classes.index(label) for label in labels if label in classes]


	# ignore
	user=inFile.strip().split('/')[-1].split('_')[0]
	name=inFile.strip().split('/')[-1].split('.')[0]
	if wordOnly:
		return name.split('_')[1:-1][0]
	if withCount and includeMan:
		return name.split('_')[1:]

	if any([label in name for label in nonManual]):
		return name.split('_')[1:-2]
	else:
		return name.split('_')[1:-1]

def getTargetPadded(targets,classes):
	maxLen=max([len(target) for target in targets])
	targets=[[classes.index(cls) for cls in target] for target in targets]
	paddedTargets=np.full((len(targets),maxLen),fill_value=(len(classes)-1),dtype='i')
	for e,l in enumerate(targets):
		paddedTargets[e,:len(l)]=l
	return paddedTargets

def getUser(inFile):
	user=inFile.strip().split('/')[-1].split('_')[0]
	name=inFile.strip().split('/')[-1]
	return user

def getDedup(k):
	k.sort()
	return list(k for k,_ in itertools.groupby(k))

def createTrainTest(userDirs,users,testCount,outFile,classes,targets=None,nonManual=None):
	classCount={}
	out=open(outFile,'w')
	for user in users:
		if targets is not None:
			for cls in targets:
				classCount[user+'-'+'-'.join(cls)]=0
		else:
			for cls in classes:
				classCount[user+'-'+'-'.join([cls])]=0
	for userDir in userDirs:
		inFiles=glob.glob(userDir+'/*')
		for inFile in inFiles:
			user=getLower(getUser(inFile))
			if targets is not None:
				labels=getLabel(inFile,nonManual)
				fromIndex=None
				if 'me' in classes:
					if any([label not in classes for label in labels if label != 'my' and label != 'i']):
						continue
				else:
					if any([label not in classes for label in labels]):
						continue
				if 'i' in labels:
					fromIndex=labels.index('i')
				elif 'my' in labels:
					fromIndex=labels.index('my')
				if fromIndex is not None:
					labels[fromIndex]='me'
				try:
					count=classCount[user+'-'+'-'.join(labels)]
				except:
					print(inFile,labels)
				if 'teach' in inFile:
					labels=['you','teachme']
				if classCount[user+'-'+'-'.join(labels)] < testCount:
					out.write("Test , {}\n".format(inFile.split('/')[-1]))
					classCount[user+'-'+'-'.join(labels)]+=1
				else:
					out.write("Train , {}\n".format(inFile.split('/')[-1]))
			else:	
				label=getLabel(inFile,wordOnly=True)
				user=getLower(inFile.strip().split('/')[-1].split('_')[0])
				name=inFile.strip().split('/')[-1].split('.')[0]
				dirName=name.split('_')[1]+name.split('_')[2].split('.')[0]
				if classCount[user+'-'+label] < testCount:
					out.write("Test , {}\n".format(user+'_'+dirName))
					classCount[user+'-'+label]+=1
				else:
					out.write("Train , {}\n".format(user+'_'+dirName))

class multiViewDatasetConcat(Dataset):
	"""
	
		filePath='./train_test_all_glosses'

		dirPath = '../data/preprocessed_data'



	"""
	# trainDataset=multiViewDatasetConcat(dirPath,classes,filePath,train=True,frameCount=10,wordOnly=True)
	def __init__(self,dirPath,classes,filePath,nonManual=None,train=True,frameCount=60,logger=None,shuffle=False,wordOnly=False):
		
		#Defining attributes

		self.dirPath=dirPath		# '../data/preprocessed_data'
		self.classes=classes		# ['why', 'help_you', 'important', 'family', 'improve', 'none', 'batangas', 'corruption', 'body', 'graduate']
		self.fileList=[]			# filenames list
		self.trainOnly=train
		self.data=None				# be filled through self.loadData(inDirs)
		self.labels=[]				# be filled through self.loadData(inDirs)
		self.frameCount=frameCount
		self.views=['xy','yz','xz']
		self.bodyParts=['body','left','right']
		self.logger=logger
		
		# self.nonManual=nonManual
		self.shuffle=shuffle			# False
		self.wordOnly=wordOnly			# True

		#Fetching Data
		f=open(filePath,'r')		# 'train_test_all_gloss'
									# txt file containing 2 comma separated words (Test/Train, filename)
		f=f.readlines()

		# Get filenames list
		if train:
			f=[f.strip().split(',')[1].strip() for f in f if 'Train' in f]
		else:
			f=[f.strip().split(',')[1].strip() for f in f if 'Test' in f]
		
		
		self.fileList=f								# filenames list
		self.fileListLow=[f.lower() for f in f]		# lowercase everything

		inDirs=glob.glob(dirPath+'/*')				# list containing the folder names in the format <user_gloss_it>
		
		# get the files in the Directory that are in the fileList
		inDirs=[inDir.replace('\\','/') for inDir in inDirs]
		inDirs=[inDir for inDir in inDirs if getLower(inDir.split('/')[-1]) in self.fileList or getLower(inDir.split('/')[-1]) in self.fileListLow]
		# print(inDirs)
		self.data,self.labels,self.inFiles=self.loadData(inDirs)
		
		self.unique_labels=getDedup(copy.deepcopy(self.labels))			# creates a new object and recursively adds the copies of nested objects present in the original elements.
		
		print([classes[label]  for label in self.unique_labels])	
		print(len(self.data),len(self.labels))
		
		shape = self.data[0].shape
		
		for i,x in enumerate(self.data):
			if x.shape!=shape:
				print("Data of different shape found",x.shape,shape,self.inFiles[i])
				exit(0)


	def __len__(self):
		return len(self.labels)
		
	def __getitem__(self,idx):
		if self.wordOnly:
			# tuple(tensor for data, tensor for label)
			return torch.tensor(self.data[idx],dtype=torch.float32),torch.tensor(self.labels[idx],dtype=torch.long)
		# if self.trainOnly:
		# 	return torch.tensor(self.data[idx],dtype=torch.float32),torch.tensor(self.labels[idx],dtype=torch.long),\
		# 	torch.tensor(self.tgtLen[idx],dtype=torch.long)
		# else:
		# 	return torch.tensor(self.data[idx],dtype=torch.float32),torch.tensor(self.labels[idx],dtype=torch.long),\
		# 	torch.tensor(self.tgtLen[idx],dtype=torch.long),self.inFiles[idx]

	def loadData(self,inDirs):
		data={'xy':[],'yz':[],'xz':[]}
		labels=[]
		files=[]
		for inDir in inDirs:									# inDirs = preprocessed folder		| inDir = folder named user_gloss_it containing  3 npy files
			npFiles=glob.glob(inDir+'/*')
			# print(npFiles)
			if len(npFiles) == 0:
				continue
			npFiles=sorted(npFiles)
			# npFiles = list that contains the files [xy,yz,xz]
			data_set=False
					
			# This is what we care about
			# np.load takes in disk file with .npy extension and returns input array
			xy=np.load(npFiles[0])	
			xz=np.load(npFiles[1])
			yz=np.load(npFiles[2])

			start=0

			if getLabel(inDir,classes=self.classes,wordOnly=True) is None:
				# This is what we care about
				"""
					if any([label not in classes for label in labels]):
						return None
				"""
				continue

			if len(xy.shape) < 4 or len(yz.shape) < 4 or len(xz.shape) < 4:
				# xy.shape = (# of frames, 74, 100, 3)
				continue

			# This is what we care about
			data['xy'].append(xy[start:start+self.frameCount])			# 1 array with N = self.frameCount elements
			data['yz'].append(yz[start:start+self.frameCount])
			data['xz'].append(xz[start:start+self.frameCount])
			
			labels.append(getLabel(inDir,classes=self.classes,wordOnly=True))

			files.append(inDir)

		
		# This is what we care about
		# data is originally a dictionary with keys: 'xy','yz','xz'
		data=[
			np.concatenate(
				(data['xy'][i],data['yz'][i],data['xz'][i]),
				axis=0
			) 
			for i in range(len(data['xy']))]

		# exclude the last 3 elements of these variables before returning
		# why lol: baka the last entries/files sa dirPath ay unnecessary so ayan
		# data=data[:-3]
		# labels=labels[:-3]
		# files=files[:-3]

		return data,labels,files


def createLogger(inDir,logFile):
	logging.basicConfig(level=logging.INFO,
		format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
		handlers=[logging.FileHandler("{0}/{1}.log".format(inDir, logFile)),logging.StreamHandler()])
	return logging.getLogger()

def computeAccuracy(labels,predictions,classes):
	return confusion_matrix(labels,predictions,labels=classes),accuracy_score(labels,predictions)

def compute_wer(ref,hyp):
	command="wer "+ref+" "+hyp
	result=run(command,stdout=PIPE,stderr=PIPE,universal_newlines=True,shell=True)
	return result.stdout.strip().split('\n')

def saveConvImages(conv,batch,saveFile):
	x=conv(batch)
	x=x.detach().cpu().numpy()
	
	for i in range(x.shape[0]):
		grid=np.zeros((200,592))
		for j in range(x.shape[1]): 
			if j<8:
				print(((j%8)*74),((j%8)*74)+74)
				grid[0:100,((j%8)*74):((j%8)*74)+74]=x[i,j,:,:].reshape(100,74)
			else:
				grid[100:200,((j%8)*74):((j%8)*74)+74]=x[i,j,:,:].reshape(100,74)
				
		im=Image.fromarray(grid)
		im=im.convert("L")
		im.save(saveFile+'-'+str(i)+'.png')
	