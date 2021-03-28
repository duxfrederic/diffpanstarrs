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


from training_set import x_train, y_train, normalize
img_rows, img_cols = 40,40
batch_size = 256
#%%

def shuffleSet(x, y, addArtificial=1):
    good = np.where(y==1)[0]
    bad = np.where(y==0)[0]
    np.random.shuffle(bad)
    where = np.append(bad[:len(good)], good)
    np.random.shuffle(where)
    where = (where,)
    x_final, y_final = x[where].copy(), y[where].copy()
    if addArtificial:
        x_art, y_art = makeModelAndLabels(20*y_final.size) 
        x_final      = x_art#np.append(x_final, x_art, axis=0)
        y_final      = y_art#np.append(y_final, y_art)
        shuffling    = np.arange(y_final.size)
        np.random.shuffle(shuffling)
        shuffling    = (shuffling, )
        x_final      = x_final[shuffling]
        y_final      = y_final[shuffling]
    return x_final, y_final


from scipy.ndimage import gaussian_filter

def makeModelAndLabels(N=100):
    shape = (img_rows, img_cols)
    x, y  = np.arange(-20,20), np.arange(-20,20)
    X, Y  = np.meshgrid(x,y)
    x_train = np.zeros((N,)+shape)
    y_train = np.zeros(N)
    for i in range(N):
        img = np.zeros(shape)
        decide = np.random.random()
        if decide > 0.95:
            n = 4
        elif decide > 0.5:
            n = 2 
        elif decide > 0.25:
            n = 1
        else: 
            n = 0
        x0_old, y0_old = 1000, 1000
        sigmaold = 0
        first = True
        for jj in range(n):
            if first or np.random.random() > 0.8: # 20% chance of having a symmetrical quasar
                sigma = abs(np.random.normal(0,2))+1
            x0, y0 = np.random.normal(0, 5, 2)
            while  (x0-x0_old)**2  + (y0-y0_old)**2 < (sigma+sigmaold)**2:
                x0, y0 = np.random.normal(0, 5, 2)
            x0_old, y0_old = x0, y0
            sigma_old = sigma
            img += np.exp(-((X-x0)**2+(Y-y0)**2)/sigma**2)*np.random.uniform(0.7, 1)
            first = False
        img += np.random.random(img.shape)*np.random.uniform(0.1, 0.3)
        
        # if np.random.random() > 0.9:
        #     if np.random.random()>0.5:
        #         img[np.random.randint(0,40), np.random.randint(0, 20):np.random.randint(20,40)] = np.random.uniform(0.5, 1)
        #     else:
        #         img[ np.random.randint(0, 20):np.random.randint(20,40), np.random.randint(0,40)] = np.random.uniform(0.5, 1)
        # if np.random.random() > 0.8:
        #    img[:, np.random.randint(0,40):] *= np.random.uniform(1.5, 2)
                
        img = gaussian_filter(img, np.random.uniform(0.2, 0.5))
                
        x_train[i] = img
        y_train[i] = 1 if n > 1 else 0
    x_train += np.random.poisson((x_train-np.min(x_train))*100)*np.random.uniform(0.001, 0.005)
    
    x_train = normalize(x_train)
    return x_train, y_train

x_test, y_test = makeModelAndLabels(20)
# for e, i in zip(x_test, y_test):
#     plt.imshow(e)
#     plt.title(i)
#     plt.waitforbuttonpress()

input_shape = (img_rows, img_cols, 1)

from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics

train_datagen = ImageDataGenerator(rescale = 1.,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range=180,
                                   vertical_flip=True,
                                   fill_mode='reflect',
                                   data_format='channels_last')

test_datagen = ImageDataGenerator()







#%%

    
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


#%%
for i in range(100):
    x, y = shuffleSet(x_train, y_train, addArtificial=1)
    x    = x.reshape(x.shape[0], img_rows, img_cols, 1)
    data_set = test_datagen.flow(x, y, batch_size=256)
    model.fit_generator(data_set, 
               samples_per_epoch=5000,
               epochs=50,
               verbose=1)
# model.fit(x_train, y_train,
          # epochs=100, verbose=1)
# model.load_weights('model.h5')
# %%
# """
x_test, y_test = makeModelAndLabels(1000)
if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    

y_predict = model.predict(x_test)

# y_predict[y_predict>0.5] = 1
# y_predict[y_predict<=0.5] = 0
# for i in range(len(y_predict)):
#     plt.imshow(x_test[i,:,:,0])
#     plt.title(f"{y_test[i]} --> {y_predict[i]}")
#     plt.waitforbuttonpress()
y_predict[y_predict>0.5] = 1
y_predict[y_predict<=0.5] = 0
print(metrics.confusion_matrix(y_test, y_predict))
#%%
y_predict = y_predict.reshape(y_predict.size)
where = np.where((y_predict == 0 ) * (y_test ==1 ) )
a = np.random.choice(where[0], size=3)
where2 = np.where((y_predict == 1 ) * (y_test == 0 ) )
b = np.random.choice(where2[0], size=3)
fig, axs = plt.subplots(2,3)
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
axs = axs.flatten()
a = np.append(a, b)
for ax, e in zip(axs, a):
    ax.axis('off')
    ax.imshow(x_test[e, :,:,0], cmap='gray')
plt.subplots_adjust(wspace=0.02, hspace=0.02)
# plt.tight_layout()
# """