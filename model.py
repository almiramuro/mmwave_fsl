from networks import getConv,getLSTM,getLinear,getDropout
import torch
import torch.nn as nn
import torch.nn.functional as F

class pointNet(nn.Module):
	def __init__(self,hidden_dim,class_size,num_layers,batch_size,dropout,use_cuda):
		super(pointNet,self).__init__()
		self.hidden_dim=hidden_dim
		self.use_cuda=use_cuda
		self.class_size=class_size
		self.modules={}
		self.lstm={}
		#self.bodyParts=['left','right','body']
		#self.partSizes={'body':150,'left':100,'right':100}
		self.modules['body']=self.localize(nn.Sequential(nn.Linear(450,400),nn.Linear(400,350),nn.Linear(350,250)))
		self.modules['left']=self.localize(nn.Sequential(nn.Linear(300,250),nn.Linear(250,200),nn.Linear(200,150)))
		self.modules['right']=self.localize(nn.Sequential(nn.Linear(300,250),nn.Linear(250,200),nn.Linear(200,150)))
		self.lstm['body']=self.localize(nn.LSTM(250,hidden_dim))
		self.lstm['left']=self.localize(nn.LSTM(150,hidden_dim))
		self.lstm['right']=self.localize(nn.LSTM(150,hidden_dim))
		self.fc1=self.localize(nn.Linear(self.hidden_dim*3,1024))
		self.drop=self.localize(nn.Dropout(p=dropout))
		self.fc2=self.localize(nn.Linear(1024,512))
		self.fc3=self.localize(nn.Linear(512,class_size))
		self.batch=batch_size
		self.num_layers=num_layers


	def init_hidden(self,batch):
		return (self.localize(torch.zeros(1,batch,self.hidden_dim)),self.localize(torch.zeros(1,batch,self.hidden_dim)))

	def forward(self,x,timeSteps):
		#self.batch=x['body'].shape[0]
		self.batch=x.shape[0]
		finalLayer=None
		for part in self.bodyParts:
			hidden=self.init_hidden(x['body'].shape[0])
			try:
				shape=x[part].shape
			except:
					logger.info("Failed to get the shape for view:%s, index: %s,shape:%s",view,t,x[views.index(view)][0].shape)
			x1=x[part].reshape(-1,timeSteps,self.partSizes[part]*3)
			x1=self.modules[part](x1)
			x1=x1.reshape(timeSteps,self.batch,-1)
			o,hidden=self.lstm[part](x1,hidden)		
			if finalLayer is None:
				finalLayer=o[-1].reshape(self.batch,-1)
			else:
				finalLayer=torch.cat((finalLayer,o[-1].reshape(self.batch,-1)),dim=1)	
			print(o[-1].shape,finalLayer.shape)
		o=self.fc3(F.relu(self.fc2(F.relu(self.fc1(finalLayer)))))
		return o.reshape(self.batch,self.class_size)


	def localize(self,x):
		if self.use_cuda:
			return x.cuda()
		else:
			return x

	def num_flat_features(self,x):
		size=x.size()[1:]
		num_features=1
		for s in size:
			num_features *=s
		return num_features

class lowResNet(nn.Module):
	def __init__(self,hidden_dim,class_size,num_layers,batch_size,dropout,use_cuda,frameCount):
		super(lowResNet,self).__init__()
		self.hidden_dim=hidden_dim
		self.use_cuda=use_cuda
		self.class_size=class_size
		self.num_layers=num_layers
		self.views=['xy','yz','xz']
		self.modules={}
		self.lstm={}
		#self.bodyParts=['body','left','right']
		#self.partSizes={'body':150,'left':100,'right':100}
		self.frameCount=frameCount
		for view in self.views:
			self.modules[view]=self.localize(nn.Sequential(nn.Conv2d(1,16,(5,5),padding=(2,2)),nn.ReLU(),nn.MaxPool2d((2,2),2),
									nn.Conv2d(16,32,(5,5),padding=(2,2)),nn.ReLU(),nn.MaxPool2d((2,2),2)))
		# for part in self.bodyParts:
		# 	self.lstm[part]=self.localize(nn.LSTM(input_size=2240*3,hidden_size=hidden_dim,num_layers=self.num_layers,batch_first=True))
		self.lstm=self.localize(nn.LSTM(input_size=2240*3,hidden_size=hidden_dim,num_layers=self.num_layers,batch_first=True))
  		self.linear=self.localize(nn.Sequential(nn.Linear(self.hidden_dim*3,1024),nn.Dropout(p=dropout),nn.Linear(1024,512),nn.Linear(512,class_size)))
		self.batch=batch_size
		self.num_layers=num_layers


	def init_hidden(self,batch):
		return (self.localize(torch.zeros(1,batch,self.hidden_dim)),self.localize(torch.zeros(1,batch,self.hidden_dim)))

	def forward(self,x):
		self.batch=x['xy'].shape[0]
		#bodyData={'body':None,'left':None,'right':None}
		Data = 'None'
		for view in self.views:
			x1=x[view].reshape(self.batch*120,1,29,40)
			x1=self.modules[view](x1)
			x1=x1.reshape(self.batch,120,-1)
   
			# for i,part in enumerate(self.bodyParts):
			# 	if bodyData[part] is None:
			# 		bodyData[part]=x1[:,(i*self.frameCount):(i*self.frameCount)+self.frameCount,:]
			# 	else:
			# 		bodyData[part]=torch.cat((bodyData[part],x1[:,(i*self.frameCount):(i*self.frameCount)+self.frameCount,:]),dim=2)

			if Data is None:
				Data = x1
			else:
				Data = torch.cat(Data, x1, dim=2) #check dim=2
   
  		finalLayer=None
    
		o,h = self.lstm(Data)
    	finalLayer=o[:,-1,:].reshape(self.batch,-1)
     
		# for part in self.bodyParts:
		# 	o,h=self.lstm[part](bodyData[part])
		# 	if finalLayer is None:
		# 		finalLayer=o[:,-1,:].reshape(self.batch,-1)
		# 	else:
		# 		finalLayer=torch.cat((finalLayer,o[:,-1,:].reshape(self.batch,-1)),dim=1)
		return self.linear(finalLayer)

	def localize(self,x):
		if self.use_cuda:
			return x.cuda()
		else:
			return x
