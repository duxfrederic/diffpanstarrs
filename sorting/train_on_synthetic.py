#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:48:22 2020

@author: frederic
"""

import numpy as np
from pandas import read_csv
from os.path import join
from glob import glob
from astropy.io import fits
""" # saved in 'training_set.csv'
source1 = "/media/frederic/01CF3B90E9BF6E00/panstarrstrials2"
source2 = "/mnt/FE867C3B867BF28F/diffpanstarrs3"
file = 'diffpanstarrs_classified.csv'
inside = 'output/*_VariabilityMap_*_normalized.fits'
test = read_csv(file)
#varimages_source = []
#%%
for e in test['filename'][1534:]:
    try:
        num = e.split('_')[0] 
        if glob(join(source2, num + '*')):
            f = glob(join(source2, num + '*'))[0]
        elif glob(join(source1, num + '*')):
            f = glob(join(source1, num + '*'))[0]
        varimg = glob(join(f, inside))[0]
        varimages_source.append(varimg)
    except:
        varimages_source.append('')
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt

img_rows, img_cols = 40,40
batch_size = 32

def makeModelAndLabels(N=1000):
    shape = (40,40)
    x, y  = np.arange(-20,20), np.arange(-20,20)
    X, Y  = np.meshgrid(x,y)
    x_train = np.zeros((N,)+shape)
    y_train = np.zeros(N)
    for i in range(N):
        img = np.zeros(shape)
        img += np.random.random(img.shape)*1.5
        decide = np.random.random()
        if decide > 0.6666:
            n = 2 
        elif decide > 0.3333:
            n = 1
        else: 
            n = 0
        for jj in range(n):
            sigma = np.random.uniform(2,6)
            x0, y0 = np.random.normal(0, 5, 2)
            img += np.exp(-((X-x0)**2+(Y-y0)**2)/sigma**2)
        x_train[i] = img
        y_train[i] = 1 if n == 2 else 0
    return x_train, y_train
#%%
# x_train, y_train = makeModelAndLabels(N=5000)
from training_set import x_train, y_train

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
train_datagen = ImageDataGenerator(rescale = 1.,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   rotation_range=15,
                                   vertical_flip=True,
                                   fill_mode='reflect',
                                   data_format='channels_last',
                                   brightness_range=[0.5, 1.5],
                                   featurewise_center=True,
                                   featurewise_std_normalization=True)

test_datagen = ImageDataGenerator()




data_set = test_datagen.flow(x_train, y_train, batch_size=batch_size)


#%%
# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)


    
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())# this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

optimizer = keras.optimizers.SGD(learning_rate=0.01)

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=optimizer,#optimizer,#keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# from sklearn.utils import class_weight
# y_ints = [y.argmax() for y in y_train]
# class_weights = class_weight.compute_class_weight('balanced',
#                                                  np.unique(y_ints),
#                                                  y_ints)
#%%

model.fit_generator(data_set, 
           samples_per_epoch=128,
           epochs=1000,
           verbose=1)
# model.fit(x_train, y_train,
          # epochs=100, verbose=1)
# model.load_weights('model.h5')
# %%
# """
x_test, y_test = makeModelAndLabels(10)
if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    

y_predict = model.predict(x_test)

# y_predict[y_predict>0.5] = 1
# y_predict[y_predict<=0.5] = 0
for i in range(len(y_predict)):
    plt.imshow(x_test[i,:,:,0])
    plt.title(f"{y_test[i]} --> {y_predict[i]}")
    plt.waitforbuttonpress()
print(metrics.confusion_matrix(y_test, y_predict))
# """