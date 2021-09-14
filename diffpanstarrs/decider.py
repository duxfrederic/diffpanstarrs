#!/usr/b3in/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:24:59 2020

@author: frederic


At first, this was supposed to sort effectively for lenses.
However it is not as simple as I anticipated, thus this will attribute a score
to the variability maps that exhibit something at all.

The higher the score, the more likely it is to be shown to a human classifier
(me) at https://diffpanstarrs.space

"""
from      os.path                         import  dirname, join
import    numpy                           as      np 
from      astropy.io                      import  fits
import    warnings 
from      astropy.utils.exceptions        import  AstropyWarning
warnings.simplefilter('ignore', AstropyWarning)

from      keras.models                    import  load_model
from      keras                           import  backend
installation_dir = dirname(__file__)
model = load_model(join(installation_dir, 'cnn_weights', 'model.hdf5'))


def normalize(images):
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



def decide(image):
    if type(image) is str:
        data = fits.getdata(image)
    else:
        data = image
    data = normalize(data)
    l, sx, sy = data.shape 
    data = data[:, sx//2-20:sx//2+20, sy//2-20:sy//2+20]
    if backend.image_data_format() == 'channels_first':
        data = data.reshape(data.shape[0], 1, 40, 40)
    else:
        data = data.reshape(data.shape[0], 40, 40, 1)
    return model.predict(data)[0][0]





















































































































