import numpy as np
import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
from string import digits
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy.spatial import ConvexHull
from scipy.special import gammainc
from  sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.stats import multivariate_normal
from sklearn.metrics import pairwise_distances
from sklearn.covariance import ledoit_wolf
import random
import copy
import math
import multiprocessing as mp
from queue import Queue
from scipy.spatial import Delaunay
import logging,sys
from scipy.stats import norm
logging.basicConfig(stream=sys.stderr,level=logging.INFO)
logger=logging.getLogger('cloudPoint')
from mpl_toolkits.mplot3d import proj3d
from matplotlib.ticker import NullLocator
import multiprocessing as mp
import io
from PIL import Image
import pandas as pd

def getTimeDiff(date1,date2):
	date1=date1/1000
	date2=date2/1000
	date1=datetime.datetime.fromtimestamp(date1)
	date2=datetime.datetime.fromtimestamp(date2)
	delta=date2-date1
	return divmod(delta.total_seconds(),60)[1]


def cloudPointVisualize(inFile,inProximity=.05,pltType='3d',saveDir=None,saveNumpy=True):
	"""
		Change inFile to the views na
	"""
	
	inFile=open(inFile,'r')
	try:
		lines=inFile.readlines()
	except:
		print("Couldn't read file {}".format(inFile))
		return

	try:
		lines=[line.strip().split(' ') for line in lines]
	except:
		print("Couldn't read file {}".format(inFile))
		return
	data_x=[]
	data_y=[]
	data_z=[]
	data_inten=[]
	clusters=[]
	axes=['xy','yz','xz']
	currTime=None
	count=0
	if saveNumpy:
		npArrays={}
		for axis in axes:
			npArrays[axis]=[]
	for line in lines:
		try:
			line=[float(x.split('::')[1]) for x in line]
		except:
			print("Couldn't read file {}".format(inFile))
			return
		if currTime is None:
			currTime=line[-1]
		elif getTimeDiff(int(currTime),int(line[-1])) > inProximity:
			if len(data_x) >0:
				data=np.array([[x,y,z] for x,y,z in zip(data_x,data_y,data_z)])
				if len(clusters) > 2:
					x=None
					for clust in clusters:
						if x is None:
							x=clust
						else:
							x=np.concatenate((x,clust),axis=0)		
					for axis in axes:
						if saveNumpy:
							npArrays[axis].append(saveFigure(x,pltTitle='Image-'+str(count),clustered=False,saveDir=saveDir+'/'+axis,pltType='2d',axis=axis,saveNumpy=saveNumpy,reSize=True))	
						else:
							saveFigure(x,pltTitle='Image-'+str(count),clustered=False,saveDir=saveDir+'/'+axis,pltType='2d',axis=axis)	
					count+=1
					clusters.pop(0)
				clusters.append(data)
			data_x=[]
			data_y=[]
			data_z=[]
			data_inten=[]
			currTime=line[-1]
		else:
			if line[0] <= 1.5 and line[0] >= 0.3 and line[1] <= 1 and line[1] >=-1 and line[2]<=1 and line[2]>=-1:
				data_x.append(line[0])
				data_y.append(line[1])
				data_z.append(line[2])
				data_inten.append(line[3])
	if saveNumpy:
		for axis in axes:
			np.save(saveDir+'/'+axis+'.npy',np.array(npArrays[axis]))

def saveFig(x, axis = None, pltTitle = None, saveDir = '.', reSize = True, saveNumpy = False):
	# saveFigure(x,pltTitle='Image-'+str(count),saveDir=saveDir+'/'+axis, axis=axis,saveNumpy=saveNumpy,reSize=True)
	
	#Change axis limits
	fig=plt.figure()
	ax=fig.add_subplot(111)
	if(axis=='xy' or axis == None):
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_xlim(-0.5, 0.8)
		ax.set_ylim(-1.2, 0.6)
	elif(axis == 'yz'):
		ax.set_xlabel('y')
		ax.set_ylabel('z')
		ax.set_xlim(-1.2, 0.6)
		ax.set_ylim(-1.4, 1.1)
	elif(axis == 'xz'):
		ax.set_xlabel('x')
		ax.set_ylabel('z')
		ax.set_xlim(-0.5, 0.8)
		ax.set_ylim(-1.4, 1.1)

	if x is None:
		plt.clf()
	else:
		ax.scatter(x[:,0],x[:,1],c='red',marker='o')

	ax.set_axis_off()
	plt.margins(0,0)
	ax.xaxis.set_major_locator(NullLocator())
	ax.yaxis.set_major_locator(NullLocator())
	plt.ioff()
	plt.axis('off')
	if reSize:
		ram=io.BytesIO()
		plt.savefig(ram,format='jpeg')
		im=Image.open(ram)
		# im=im.convert('1')
		im=im.resize((100,74),Image.LANCZOS)	
		if saveNumpy:
			ram.close()
			plt.close()
			return np.asarray(im)
		# print(saveDir)
		# exit()
		im.save(saveDir+'/'+pltTitle+'.png')
		ram.close()
		plt.close()
	else:
		plt.savefig(saveDir+'/'+pltTitle+'.jpeg')
		plt.close()
	

def toFloat(pair = [None, None]):
	out = []
	if len(pair) > 2:
		count = len(pair) - 2
		while count:
			if '\r' in pair:
				pair.remove('\r')    
			else:
				pair.remove('')
			count -= 1
	for i in [0,1]:
		if 'e' in pair[i]:
			a = pair[i].split('e')
			if a[-1][0] == '+':
				out.append(float(a[0])*(10^(int(a[-1][1::]))))
			elif a[-1][0] == '-':
				out.append(float(a[0])/(10**(int(a[-1][1::]))))
		else:
			out.append(float(pair[i]))        
	out = np.array(out, dtype=float)
	return out
			
def npArray(data, view = None):
	out = []
	for i in range (0, data.size):
		frame = []
		for j in range (0, len(data[i].split(' \n '))):
			array = np.array(data[i].split(' \n ')[j].split('\n '))
			array2d = []
			for k in range (0, array.size):
				array[k] = array[k].replace('[', '')
				array[k] = array[k].replace(']', '')
				# print(array[k].split(' '))
				array2d.append(toFloat(array[k].split(' ')))
			# exit()                    
			frame.append(np.array(array2d))
		out.append(np.array(frame, dtype=np.ndarray))
	# print(out[0].size)
	# out = np.array(out, dtype=np.ndarray)
	# print(out)
	return out

def main():
	filePath = 'data/preprocessed_data'
	saveDir = 'data/images'
	if not os.path.isdir(saveDir):
		os.makedirs(saveDir)
	signers = ['aaron', 'mira', 'luis']
	signs = ['why', 'help_you', 'important', 'family', 'improve', 'none', 'batangas', 'corruption', 'body', 'graduate']
	dataset = []
	for signer in signers:
		for sign in signs:
			sample = 15
			while sample:
				dataset.append('{}_{}_{}_processed.csv'.format(signer, sign, sample))
				sample -= 1
	for data in dataset:
		file = filePath + '/' + data
		d = pd.read_csv(file)
		views = {'xy': [], 'yz': [], 'xz': []}
		views['xy'] = (npArray(d.xy.values, 'xy'))
		views['yz'] = (npArray(d.yz.values, 'yz'))
		views['xz'] = (npArray(d.xz.values, 'xz'))
		for view in views:
			for frame in range (0, len(views[view])):
				# print(len(views[view][frame][0]))
				saveFig(views[view][frame][0], axis=view, pltTitle = data[:-13:]+view+'_'+str(frame), saveDir = saveDir, reSize = True, saveNumpy = False)
		# exit()
	# file = 'aaron_hello_world_2_processed.csv'
	# d = pd.read_csv(file)
	# views = {'xy': [], 'yz': [], 'xz': []}
	# views['xy'] = npArray(d.xy.values, 'xy')
	# views['yz'] = npArray(d.yz.values, 'yz')
	# views['xz'] = npArray(d.xz.values, 'xz')
	# for view in views:
	#     for frame in range (0, len(views[view])):
	#         saveFig(views[view][frame], axis=view, pltTitle = file[:-13:]+view, saveDir = saveDir, reSize = True, saveNumpy = False)
			
if __name__ == '__main__':
	main()