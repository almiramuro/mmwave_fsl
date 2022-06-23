import torch
import torch.nn as nn
import torch.nn.functional as F


class wordNet(nn.Module):
	def __init__(self,hidden_dim,class_size,num_layers,batch_size,dropout,use_cuda,frameCount,dataParallel=False):
		super(wordNet,self).__init__()
		self.hidden_dim=hidden_dim		#2048
		self.use_cuda=use_cuda			#True
		self.class_size=class_size		#10 glosses
		self.num_layers=num_layers		#2
		self.views=['xy','yz','xz']
		self.modules={}
		self.lstm={}
		self.frameCount=frameCount		#10
		self.dataParallel=dataParallel	#True

		# CNN 
		for view in self.views:
			if self.dataParallel:
				self.modules[view]=nn.DataParallel(self.localize(
					nn.Sequential(
						nn.Conv2d(3,16,(5,5),padding=(2,2)),
						nn.ReLU(),
						nn.MaxPool2d((2,2),2),
						nn.Conv2d(16,32,(5,5),padding=(2,2)),
						nn.ReLU(),
						nn.MaxPool2d((2,2),2),
						nn.Conv2d(32,64,(5,5),padding=(2,2)),
						nn.ReLU(),
						nn.MaxPool2d((2,2),2),
						nn.Conv2d(64,128,(5,5),padding=(2,2)),
						nn.ReLU(),
						nn.MaxPool2d((2,2),2)
					)))
			else:
				self.modules[view]=self.localize(
					nn.Sequential(
						nn.Conv2d(3,16,(5,5),padding=(2,2)),
						nn.ReLU(),
						nn.MaxPool2d((2,2),2),
						nn.Conv2d(16,32,(5,5),padding=(2,2)),
						nn.ReLU(),
						nn.MaxPool2d((2,2),2),
						nn.Conv2d(32,64,(5,5),padding=(2,2)),
						nn.ReLU(),
						nn.MaxPool2d((2,2),2),
						nn.Conv2d(64,128,(5,5),padding=(2,2)),
						nn.ReLU(),
						nn.MaxPool2d((2,2),2)
					))

		# LSTM
		if self.dataParallel:
			self.lstm=nn.DataParallel(self.localize(
				nn.LSTM(input_size=3072*3,hidden_size=hidden_dim,num_layers=self.num_layers,batch_first=True,bidirectional=True)
				))
			self.linear=nn.DataParallel(self.localize(
				nn.Sequential(
					nn.Linear(self.hidden_dim*2,self.hidden_dim),
					nn.ReLU(),
					nn.Linear(self.hidden_dim,1024),
					nn.ReLU(),nn.Dropout(p=dropout),
					nn.Linear(1024,512),
					nn.ReLU(),
					nn.Linear(512,class_size)
				)))
		else:
			self.lstm=self.localize(
				nn.LSTM(input_size=3072*3,hidden_size=hidden_dim,num_layers=self.num_layers,batch_first=True,bidirectional=True)
			)
			self.linear=self.localize(
				nn.Sequential(
					nn.Linear(self.hidden_dim*2,self.hidden_dim),
					nn.ReLU(),
					nn.Linear(self.hidden_dim,1024),
					nn.ReLU(),
					nn.Dropout(p=dropout),
					nn.Linear(1024,512),
					nn.ReLU(),
					nn.Linear(512,class_size)
				))

		# Out
		# i believe ito lang ung self.linear with expected numbers
		self.linear_frame=self.localize(
			nn.Sequential(
				nn.Linear(self.hidden_dim,1024),
				nn.ReLU(),
				nn.Dropout(p=dropout),
				nn.Linear(1024,512),
				nn.ReLU(),
				nn.Sequential(
					nn.Linear(512,60)
				)
			))
			
		self.batch=batch_size		# 10
		self.num_layers=num_layers
		self.directions=2


	def init_hidden(self,batch):
		# idk when/where is this called
		return (self.localize(torch.zeros(self.num_layers*self.directions,batch,self.hidden_dim)),
				self.localize(torch.zeros(self.num_layers*self.directions,batch,self.hidden_dim)))

	def forward(self,x,t):
		# from line 119, x is 5 dimensional (input tensor)
		self.batch=x.shape[0]		# len of the first dimension tensor (torch.Size([1stdimlen,2nddimlen,3rddimlen,4thdimlen]))
		# Input Layer
		self.t=t					# hmm what purpose?
		bodyData={'body':None,'left':None,'right':None}
		finalLayer=None
		for i,view in enumerate(self.views):
			# [(0, 'xy'), (1, 'yz'), (2, 'xz')]
			x1=x[:,i:i+self.frameCount,:,:,:].reshape(self.batch*self.frameCount,3,74,100)
			# x[...] Get all of the first dimension
			# x[...] Get the second dimension's i:i+self.frameCount elements and everything inside it ([:,:,:])
			# x[...] is still a 5dimensional tensor
			# x[...] everything in the view, and there are 40 frames per view
			# x1 reshaped into a 4dimensional tensor with size (self.batch*40, 3, 74, 100)
			# self.batch*40 =  
			# 3 = first Conv2d in_channels
			# 74 = ?
			# 100 = ?

			x1=self.modules[view](x1)						# undergo the sequential model forwarding CNN
			x1=x1.reshape(self.batch,self.frameCount,-1)	# reshape to 3D tensor with shape (self.batch, self.frameCount, everything else concatenated into 1 tensor)
			if finalLayer is None:
				finalLayer=x1								# processed xy view 
			else:
				finalLayer=torch.cat((finalLayer,x1),dim=2)	# Concatenates the given sequence of seq tensors (finalLayer,x1) in the given dimension. 
															# basically concatenates all the processed views into a single 3D tensor with the same first two dimension lengths (self.batch,self.frameCount)
		
		# undergo the lstm
		o,h=self.lstm(finalLayer)							# o = output tensor, h=tuple of tensors
		o1=o[:,-1,:].reshape(self.batch,-1)					#
		# o[...] Get all of the first dimension
		# x[...] Get the last elements of the second dimension and everything inside it ([:])
		# x[...] is then a 2D tensor
		# reshape to 2D sensor with # of rows = self.batch 
		
		# ready for linear layer
		o1=self.linear(o1)									# 
		return o1	

	def localize(self,x):
		if self.use_cuda:
			return x.cuda()
		else:
			return x
