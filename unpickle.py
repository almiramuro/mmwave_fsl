import pickle
import numpy as np

filename = 'raw_2022-04-30_16-20-25.pkl'
with open(filename, 'rb') as handle:
    b = pickle.load(handle)

# print(b['1651302264453'].shape)
print(b)

for key in b:
    print(b[key])