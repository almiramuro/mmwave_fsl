import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import glob
import torch.optim as optim
from string import digits
import logging
import logging.handlers

class Net(nn.Module):
	def __init__(self, class_size, hidden_dim=2048, num_layers=2, dropout=0.65, frameCount=40, device=torch.device('cuda')):
		super(Net, self).__init__()
		self.class_size = class_size
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.dropout = dropout
		self.frameCount = frameCount
		self.device = device

		self.views = ['xy', 'yz', 'xz']
		self.conv = {}

		for view in self.views:
			self.conv[view] = nn.Sequential(
				nn.Conv2d(1, 16, 5, padding=(2,2)), nn.ReLU(), nn.MaxPool2d((2,2), 2),
				nn.Conv2d(16, 32, 5, padding=(2,2)), nn.ReLU(), nn.MaxPool2d((2,2), 2),
				nn.Conv2d(32, 64, 5, padding=(2,2)), nn.ReLU(), nn.MaxPool2d((2,2), 2),
				nn.Conv2d(64, 128, 5, padding=(2,2)), nn.ReLU(), nn.MaxPool2d((2,2), 2)
			)

		self.lstm = nn.LSTM(
			input_size=768, 
			hidden_size=self.hidden_dim, 
			num_layers=self.num_layers, 
			batch_first=True)
			# bidirectional=False)
		self.dense = nn.Sequential(
			nn.Linear(self.hidden_dim, 1024),
			nn.Linear(1024, 512),
			nn.Dropout(p=dropout),
			nn.Linear(512, class_size)
		)

	def forward(self, x):
		self.batch = x['xy'].shape[0]
		# print(x['xy'].shape[0])
		data = None
		# exit()
		for view in self.views:
			x1 = x[view].reshape(self.batch*self.frameCount, 1, 29, 40) # self.batch*# of frames, 1 channel, (29 rows, 40 columns) - 29x40 image
			x1 = self.conv[view](x1)
			x1 = x1.reshape(self.batch, self.frameCount, -1)
			# print(x1.shape)
			if data is None:
				data = x1
			else:
				data = torch.cat((data, x1), dim=2)
		data = data.to(self.device)
		# exit()
		# print('Passed thru multiview cnn')
		
		o, _ = self.lstm(data)
		# print('Passed thru lstm')
		final_layer = o[:,-1,:].reshape(self.batch, -1)
		
		return self.dense(final_layer)

# class DLModule(nn.Module):
#     def __init__(self, in_planes = 5 , out_planes = 10, kernel_size = (2,2,2), padding = 0):
#         super().__init__()
#         self.conv1 = nn.Conv3d()
#         self.bn1 = nn.BatchNorm3d()
#         self.conv2 = nn.Conv3d()
#         self.conv3 = nn.Conv3d()
        
#         self.fc1 = nn.Linear()
#         self.fc2 = nn.Linear()
#         self.drop = nn.Dropout3d(p = 0.5, inplace = False)
#         self.fc3 = nn.Linear()

#         self.lstm = nn.LSTM()
        
#         # LSTM

#         # 2 Linear
#         # torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

       

    
#     def forward(self, x):
#         y = nn.ReLU()(self.conv1(x))
#         y = nn.MaxPool3d(kernel_size=2)(y)
#         y = nn.ReLU()(self.conv2(y))
#         y = nn.MaxPool3d(kernel_size=2)(y)
#         y = nn.ReLU()(self.conv3(y))
#         y = rearrange(y, 'b c h w -> b (c h w)')        # print shape (this will print 2048)

#         y = self.fc1(y)
#         return y


# nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# (Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape))
"""
    in_channels (int) – Number of channels in the input image
        # frames
    out_channels (int) – Number of channels produced by the convolution
        # frames
    kernel_size (int or tuple) – Size of the convolving kernel
        # 2,2,2
    stride (int or tuple, optional) – Stride of the convolution. Default: 1

    padding (int, tuple or str, optional) – Padding added to all six sides of the input. Default: 0

    padding_mode (string, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'

    dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1

    groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1

    bias (bool, optional) – If True, adds a learnable bias to the output. Default: True

    input = (N,Ci,D,H,W)
    output = (N,Co,Do,Ho,Wo)
    N = len(dataset) = 
    C = num_features = (color)
"""

