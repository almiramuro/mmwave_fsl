import multiprocessing as mp
import glob
from torch.utils.data import Dataset
from string import digits
import numpy as np
import torch
import logging 
from sklearn.metrics import accuracy_score,confusion_matrix

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

def getLabel(inFile,classes=None,withCount=True):
	if classes is not None:
		return classes[[cls in inFile for cls in classes].index(True)]
	user=inFile.strip().split('/')[-1].split('_')[0]
	name=inFile.strip().split('/')[-1]
	dirName=name.split('_')[1]+name.split('_')[2].split('.')[0]
	if withCount:
		return dirName
	else:
		return rmDigit(dirName)

def getUser(inFile):
	user=inFile.strip().split('/')[-1].split('_')[0]
	name=inFile.strip().split('/')[-1]
	return user

def createTrainTest(userDirs,users,testCount,outFile,classes):
	classCount={}
	out=open(outFile,'w')
	for user in users:
		for cls in classes:
			classCount[user+'-'+cls]=0
	for userDir in userDirs:
		inFiles=glob.glob(userDir+'/*')
		for inFile in inFiles:
			user=getLower(getUser(inFile))
			label=getLabel(inFile)
			if label not in classes:
				continue
			if classCount[user+'-'+label] < testCount:
				out.write("Test , {}\n".format(inFile.split('/')[-1]))
				classCount[user+'-'+label]+=1
			else:
				out.write("Train , {}\n".format(inFile.split('/')[-1]))

class multiViewDataset(Dataset):

	def __init__(self,dirPath,classes,filePath,train=True,frameCount=40,logger=None):
		self.dirPath=dirPath
		self.classes=classes
		self.fileList=[]
		self.trainOnly=train
		self.data=None
		self.labels=[]
		self.frameCount=frameCount
		self.views=['xy','yz','xz']
		self.bodyParts=['body','left','right']
		self.logger=logger
		f=open(filePath,'r')
		f=f.readlines()
		if train:
			f=[f.strip().split(',')[1] for f in f if 'Train' in f]
		else:
			f=[f.strip().split(',')[1] for f in f if 'Test' in f]
		f=[getUser(f)+'_'+getLabel(f) for f in f]
		self.fileList=f
		self.fileListLow=[f.lower() for f in f]
		
		inDirs=glob.glob(dirPath+'/*')
		inDirs=[inDir for inDir in inDirs if inDir.split('/')[-1] in self.fileList or inDir.split('/')[-1] in self.fileListLow]
		self.data,self.labels=self.loadData(inDirs)

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


def createLogger(inDir,logFile):
	logging.basicConfig(level=logging.INFO,
		format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
		handlers=[logging.FileHandler("{0}/{1}.log".format(inDir, logFile)),logging.StreamHandler()])
	return logging.getLogger()

def computeAccuracy(labels,predictions,classes):
	return confusion_matrix(labels,predictions,classes),accuracy_score(labels,predictions)