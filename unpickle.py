import pickle
import numpy as np

filename = 'raw_2022-05-13_02-44-09.pkl'
with open(filename, 'rb') as handle:
    b = pickle.load(handle)

# print(b['1651302264453'].shape)
print(b)

for key in b:
    print(b[key])