import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import regularizers
from keras import backend as keras


class UNet(Model):
    def __init__(
            self,
            input_size=(64, 64, 3),
    ):
        x = Input(shape=input_size, name='x')
        b = Input(shape=(*input_size[:-1], 1), name='b')

        x_o = Multiply()([x, b])
        #x_o = x

        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x_o)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        drop2 = Dropout(0.5)(conv2)

        up3 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2))
        merge3 = concatenate([conv1,up3], axis = 3)
        conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge3)
        conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        conv3 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3) 
        conv4 = Conv2D(3, 1, activation = 'softmax')(conv3)

        outputs = conv4
        inputs = {"x": x, "b": b}

        super().__init__(inputs, outputs)
