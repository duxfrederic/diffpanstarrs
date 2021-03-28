# -*- coding: utf-8 -*-
# we use our local generated variability maps together with
# the results from the website at diffpanstarrs.space  

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
varimages_source = []
# %%
for e in test['filename']:
    try:
        num = e.split('_')[0] 
        if glob(join(source2, num + '*')):
            f = glob(join(source2, num + '*'))[0]
        elif glob(join(source1, num + '*')):
            f = glob(join(source1, num + '*'))[0]
        else:
            varimages_source.append('')
            continue
        varimg = glob(join(f, inside))[0]
        varimages_source.append(varimg)
    except:
        varimages_source.append('')
# """
#%%
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt

batch_size = 50
num_classes = 1
epochs = 12

# input image dimensions
img_rows, img_cols = 40, 40

#%%
""" # saved in x_train.npy and y_train.npy
data =  read_csv('training_set.csv').dropna()
img_rows, img_cols = 40, 40
x_train = np.zeros((len(data), img_rows, img_cols), dtype=np.float32)
y_train = np.zeros(len(data))
for i, (f, r) in enumerate( zip(data['variability_paths'], data['Lens']) ):
    x_train[i] = fits.getdata(f)
    y_train[i] = 1 if r > 0 else 0 
#%%
# """
#%%

def normalize(images):
    # m, s = x_train.mean(axis=(-2,-1)), x_train.std(axis=(-2,-1))
    images[np.isnan(images)] = 0
    if len(images.shape) == 2:
        images = np.array([images])
    for i in range(len(images)):
        m, s = np.mean(images[i]), np.std(images[i])
        N = 10
        images[i][images[i]>m+N*s] = m+N*s
        images[i] -= np.min(images[i])
        images[i] /= np.max(images[i])
    return images



x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
y_train = y_train.astype(float)
x_train = x_train.astype(float)
x_train = normalize(x_train)
# good = np.where(y_train==1)[0]
# bad = np.where(y_train==0)[0]
# np.random.shuffle(bad)

# where = np.append(bad[:len(good)], good)
# np.random.shuffle(where)
# where = (where,)
# x_train, y_train = x_train[where], y_train[where]





#%%
# from sklearn import datasets, svm, metrics
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# X_train = x_train.reshape((len(y_train), -1))
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)
#%%
# classifier = LogisticRegression()

# classifier.fit(X_train, y_train)

# predicted = classifier.predict(X_test)
# print(predicted)
#%%



