from unet import *
import pickle
import numpy as np
import tensorflow as tf

from unet import *
from tqdm import tqdm

# Load the data
with open("retina_data.pkl", "rb") as f:
    data = pickle.load(f)

model = unet(input_size=(128, 128, 1))

"""
masks = 5

train_data = data['train'][0].astype(np.float32) / 255.
masked_train = train_data[0] * np.random.choice([0, 1], size=train_data[0].shape[0]**2, p=[0.6, 0.4]).reshape(train_data[0].shape[0], train_data[0].shape[0], 1)
masked_train = masked_train.reshape(1, 128, 128, 3)

valid_data = data['valid'][0].astype(np.float32) / 255.
masked_valid = valid_data[0] * np.random.choice([0, 1], size=valid_data[0].shape[0]**2, p=[0.6, 0.4]).reshape(valid_data[0].shape[0], valid_data[0].shape[0], 1)
masked_valid = masked_valid.reshape(1, 128, 128, 3)

for i in tqdm(range(train_data.shape[0])):
    for j in range(masks):
        if not(i == 0 and j == 0): 
            temp_train = train_data[i] * np.random.choice([0, 1], size=train_data[i].shape[0]**2, p=[0.6, 0.4]).reshape(train_data[i].shape[0], train_data[i].shape[0], 1) 
            masked_train = np.concatenate((masked_train, temp_train.reshape(1, 128, 128, 3)), axis=0)

for i in tqdm(range(valid_data.shape[0])):
    for j in range(masks):
        if not(i == 0 and j == 0): 
            temp_valid = valid_data[i] * np.random.choice([0, 1], size=valid_data[i].shape[0]**2, p=[0.6, 0.4]).reshape(valid_data[i].shape[0], valid_data[i].shape[0], 1) 
            masked_valid = np.concatenate((masked_valid, temp_valid.reshape(1, 128, 128, 3)), axis=0)
"""

model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
model.fit(data['train'][0].astype(np.float32)/255., data['train'][1].astype(np.float32)/255., batch_size=32, epochs=10, validation_data=(data['valid'][0].astype(np.float32)/255., data['valid'][1].astype(np.float32)/255.))
