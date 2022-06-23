import numpy as np

if __name__=="__main__":
    users = ['aaron', 'mira', 'luis']
    classes = ['why', 'help_you', 'important', 'family', 'improve', 'none', 'batangas', 'corruption', 'body', 'graduate']

    out = []

    for name in users:
        for gloss in classes:
            for i in range(12):
                out.append('Train,'+name+'_'+gloss+'_'+str(i+1))
            for i in range(12,15):
                out.append('Test,'+name+'_'+gloss+'_'+str(i+1))
    
    print(len(out))
    with open('train_test_all_glosses','w') as f:
        for element in out:
            f.write('%s\n'%element)

    f=open('train_test_all_glosses','r')		# 'train_test_words_all'
									# txt file containing 2 comma separated words (Test/Train, filename)
    f=f.readlines()

    # Get filenames list
    f_train=[f.strip().split(',')[1].strip() for f in f if 'Train' in f]
    f_test=[f.strip().split(',')[1].strip() for f in f if 'Test' in f]

    print(f_train)
    print(f_test)
