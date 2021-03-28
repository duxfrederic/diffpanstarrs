#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
sys.path.append("..")

from    astropy.io            import  fits
from    scipy.stats           import  shapiro, probplot
import  matplotlib.pyplot     as      plt


from    diffpanstarrs.wavelet import  waveletDenoise 


image = fits.open('test_data/test_image_moments/HE0435-1223_VariabilityMap_r.fits')
image = image[0].data



imageden  = waveletDenoise(image, weight=0.2)
residuals = imageden - image



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)

ax1.imshow(image, origin='lower')
ax2.imshow(imageden, origin='lower')
ax3.imshow(residuals, origin='lower')

ax1.set_title('image')
ax2.set_title('denoised')
ax3.set_title('difference')

#%%
plt.figure()
probplot(residuals.flatten(), plot=plt)
# normaltest = shapiro(residuals.flatten())