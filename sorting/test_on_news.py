#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 21:29:06 2020

@author: frederic
"""
#%%
from os.path import join, basename
from glob import glob
path = "/mnt/FE867C3B867BF28F/diffpanstarrs3"
files = [glob(join(path, f"{i:0>5}_*"))[0] for i in range(5880, 5881)]

#%%
from keras.models import load_model

model = load_model('newmodel.hdf5')
img_rows, img_cols = 40, 40
#%%
import numpy as np
from astropy.io import fits
from training_set import normalize
# datas = []
# img_rows, img_cols = 40, 40
# for file in files:
#     num = basename(file).split('_')[0]
#     try:
#         image = glob(join(file, 'output', f'{num}*VariabilityMap_ri_normalized.fits'))[0]
#     except IndexError:
#         continue
#     data = fits.getdata(image)
#     data = normalize(data)
#     # data = data.reshape(1, img_rows, img_cols, 1)
#     datas.append(data)
#     # input_shape = (img_rows, img_cols, 1)
#     # plt.imshow(data[0,:,:,0])
#     # plt.title(f"{num} --> {model.predict(data)[0,0]:.02f}")
#     # plt.waitforbuttonpress()
# datas = np.array(datas)
# datas = datas.reshape(len(datas), img_rows, img_cols, 1)

# y_pred = model.predict(datas)
# #%%
# where = np.where(y_pred>0.5)

#%%
# import matplotlib.pyplot as plt
# for e in where[0]:
#     plt.imshow(datas[e, :, :, 0])
#     plt.title(e)
    # plt.waitforbuttonpress()
    
    
#%%

y_test, x_test = np.load('y_test.npy'), np.load('x_test.npy')
x_test = normalize(x_test)
x_test_easyshape = x_test.copy()
x_test = x_test.reshape(len(x_test), img_rows, img_cols, 1)
y_pred = model.predict(x_test)
y_pred_cont = y_pred.copy()
y_pred[y_pred>0.5]  = 1
y_pred[y_pred<=0.5] = 0


from sklearn.metrics import confusion_matrix 

print(confusion_matrix(y_test, y_pred))
y_pred = y_pred[:,0]
#%%
where = np.where((y_test == 1) * (y_pred == 0))
import matplotlib.pyplot as plt
i = 0
for im, e in zip(x_test_easyshape[where], y_pred_cont[where]):
    plt.imshow(im, origin='lower')
    plt.title(e)
    plt.waitforbuttonpress()
    i+=1


#%%
images = []
y_test = []
y_pred = []
names = []
for e in glob('/media/frederic/kingston_data/okay_good_lenses/*fits'):
    name = basename(e).split('_')[0]
    names.append(name)
    image = fits.getdata(e)
    images.append(image.copy())
    human = int(basename(e).split('_')[-1].replace('.fits', ''))
    y_test.append(human)
    image[np.isnan(image)] = 0
    image = normalize(image)
    image = image.reshape(1, 40, 40, 1)
    y_pred1 = model.predict(np.flip(image, axis=1))[0][0]
    y_pred2 = model.predict(np.flip(image, axis=2))[0][0]
    y_pred3 = model.predict(np.flip(image))[0][0]
    
    y_pred.append(1/3*(y_pred1+y_pred2+y_pred3))

images = np.array(images)
names = np.array(names)
y_test = np.array(y_test)
y_pred = np.array(y_pred)
y_pred_full = y_pred.copy()
y_pred[y_pred>0.5] = 1
y_pred[y_pred<=0.5] = 0

print(confusion_matrix(y_test, y_pred))

where1 = np.where((y_pred == 1) * (y_test == 0))
#%%
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size'   : 10})
rc('text', usetex=True)
fontsize = 10
fig, axs = plt.subplots(3, 7, figsize=(7.2455, 3.105))
axs = axs.flatten()
axs[14], axs[-3] = axs[-3], axs[14]
for ax, im, s, name in zip(axs, images[where1], y_pred_full[where1], names[where1]):
    ax.imshow(im, cmap='gray', origin='lower')
    ax.text(0.1, 30, f"{s:.02f}", color='white', fontsize=fontsize)
    t = ax.text(2.5, 4, name, color='white', fontsize=4.6)
    t.set_bbox(dict(facecolor='black', alpha=0.6, edgecolor='None', linewidth=0))
for ax in axs:
    ax.axis('off')
plt.subplots_adjust(wspace=0.02, hspace=0.02)
plt.savefig('/home/frederic/Nextcloud/diffpanstarrs/docs/report2/false_positives.pdf', bbox_inches='tight',
            pad_inches=0)
    
#%%
where2 = np.where((y_pred == 0) * (y_test == 1))
fig, axs = plt.subplots(2, 7, figsize=(7.2455,2.07))
axs = axs.flatten()
for ax, im, s, name in zip(axs, images[where2], y_pred_full[where2], names[where2]):
    ax.imshow(im, cmap='gray', origin='lower')
    ax.text(0.1, 30, f"{s:.02f}", color='white', fontsize=fontsize)
    t = ax.text(2.5, 4, name, color='white', fontsize=4.6)
    t.set_bbox(dict(facecolor='black', alpha=0.6, edgecolor='None', linewidth=0))
for ax in axs:
    ax.axis('off')
plt.subplots_adjust(wspace=0.02, hspace=0.02)
plt.savefig('/home/frederic/Nextcloud/diffpanstarrs/docs/report2/missed.pdf', bbox_inches='tight',
            pad_inches=0)
    