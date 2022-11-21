import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import pickle

# Load the retina data
dirname = './retina_filter'

# giving file extension
ext = ('png')
I = plt.imread(os.path.join(dirname, 'im0001.net.' + ext))
I = I.astype(np.float32)
I = I.flatten()
I = I.reshape(1, -1)
 
# iterating over all files
for files in os.listdir(dirname):
    if files.endswith(ext) and not files.startswith('im0001.net'):
        temp = plt.imread(os.path.join(dirname, files))
        temp = temp.astype(np.float32)
        temp = temp.flatten()
        temp = temp.reshape(1, -1)
        I = np.concatenate((I, temp), axis=0)
    else:
        continue

train_indices = np.arange(397-2*48)
valid_indices = np.arange(397-2*48, 397-48)
test_indices = np.arange(397-48, 397)

data = {
        'train': I[train_indices,:],
        'valid': I[valid_indices,:],
        'test': I[test_indices,:]
        }
print(data)

with open('retina_data.pkl', 'wb') as f:
    pickle.dump(data, f)
