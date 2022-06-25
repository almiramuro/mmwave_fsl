import numpy as np

if __name__=="__main__":
	setups = ['indoor','indoor_wnoise','outdoor']
	users = ['alma', 'melissa', 'rommel']
	classes = open('../dl_model/glosses','r',encoding='utf-8-sig').readlines()
	classes = [ gloss.strip() for gloss in classes ]
	combined=True
	
	out = []

	if(combined):
		for venue in setups:
			for name in users:
				for gloss in classes:
					for i in range(15):
						num = '0'+str(i+1) if(i < 9) else str(i+1)
						mode = 'Train' if(i < 12) else 'Test'
						filename = '_'.join([name,gloss,num+'.pkl'])
						path = '/'.join([venue,filename])
						out.append(','.join([mode,path]))
	
	print(len(out))
	with open('../dl_model/combined_train_test_all_glosses','w') as f:
		f.writelines('\n'.join(out))

	# f=open('train_test_all_glosses','r')		# 'train_test_words_all'
	# 								# txt file containing 2 comma separated words (Test/Train, filename)
	# f=f.readlines()

	# # Get filenames list
	# f_train=[f.strip().split(',')[1].strip() for f in f if 'Train' in f]
	# f_test=[f.strip().split(',')[1].strip() for f in f if 'Test' in f]

	# print(f_train)
	# print(f_test)
