from utility import createLogger,multiViewDatasetConcat,computeAccuracy
from torch.utils.data import DataLoader
from model import wordNet
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F
import sys
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

if __name__=="__main__":
	"""
        run the file: test.py datasetfolder modelname

        e.g.
        Enter in cmd line:
            python test.py outdoor_24_train_test_all_glosses latest-model.pth
		to signify:
			- use the filePath outdoor_24_train_test_all_glosses
			- use modelpath latest-model.pth

    """
	
	setup, _model= sys.argv[1], sys.argv[2]
	
	
	# users = ['aaron', 'mira', 'luis']
	
	_24classes = ['hello', 'nice_meet_you', 'good_night', 'sign_name', 'how_you', 'why', 'sleep', 'calendar', 'cook', 'computer', 'help_you', 'important', 'family', 'improve', 'none', 'batangas', 'bulacan', 'bicol', 'flood', 'go_home', 'corruption', 'body', 'life', 'graduate']
	_10classes = ['why', 'help_you', 'important', 'family', 'improve', 'none', 'batangas', 'corruption', 'body', 'graduate']
	
	classes = _10classes if(setup.split('_')[1] == '10') else _24classes
	filePath = '_'.join([setup.split('_')[i] for i in range(2)]) + '_train_test_all_glosses'
	dirPath = '../data/preprocessed_data/'+setup

	testDataset=multiViewDatasetConcat(dirPath,classes,filePath,train=False,frameCount=10,wordOnly=True)

	torch.manual_seed(1)
	torch.cuda.manual_seed(1)
	np.random.seed(1)
	torch.backends.cudnn.deterministic = True

	net=wordNet(2048,len(classes),2,5,0.65,False,10,True)
	modelPath='./'+_model
	net.load_state_dict(torch.load(modelPath,map_location='cpu'),strict=False)
	m=nn.Softmax(dim=1)
	predictions=[]
	labels=[]
	net.eval()
	multiViewDataLoader=DataLoader(testDataset,5,shuffle=False)
	count = 0
	for x,label in multiViewDataLoader:
		print('sample ', count)		
		o=net(x,10)	
		prediction = torch.max(m(o),dim=1)[1].cpu().numpy().tolist()
		predictions += prediction
		labels+=label.cpu().numpy().tolist()
		print(prediction, label)
		count += 1
	confusion,accuracy=computeAccuracy(labels,predictions,[i for i in range(len(classes))])
	print("The accuracy for %s using %s is: %f"%(setup,_model,accuracy))
	print("The confusion Matrix is")
	print(confusion)
	ConfusionMatrixDisplay.from_predictions(labels,predictions)
	figFile = 'testfile-%s_and_model-%s'%(setup,_model[:-4])
	plt.savefig(figFile + '.png')