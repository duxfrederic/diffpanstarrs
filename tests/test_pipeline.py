#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:11:39 2020

@author: frederic
"""
from os.path import join, exists, basename
from os import mkdir, remove, makedirs
from glob import glob
from shutil import rmtree
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import PercentileInterval, SqrtStretch
import traceback
from PIL import Image
import io
from diffpanstarrs import downloadAndProcess, config
config.magnitude_threshold_star = 16.0
from diffpanstarrs.decider import decide

workdir = "/mnt/FE867C3B867BF28F/diffpanstarrs3"

catalogue = "test_data/news_catalogue/extended_reduced2.fits"
data = fits.getdata(catalogue)
# test with the 100 brightest sources, should take ~15GB with 512x512 images
RA, DEC = data['WISE_RA'], data['WISE_DEC']
#%%

import logging
logging.basicConfig(filename='RESULTS.log',level=logging.INFO, format='%(message)s')
logger = logging.getLogger('RESULTS.log')
logger.setLevel(logging.INFO)


def deleteFiles(specworkdir):
    originals = glob(join(specworkdir, 'rings*'))
    for e in originals:
        remove(e)
    diffimgs = glob(join(specworkdir, 'output', 'rings*'))
    for e in diffimgs:
        remove(e)
    rmtree(join(specworkdir, 'output', 'Sextraction'))

#%%
def saveVariabilityMap(varpath, normpath, bigvarpath, origpath, score, outdir=None):
    savename = varpath.replace('.fits', '.png')
    savename = basename(savename)
    id_, ra, dec, var, channels = savename.split('_')
    savename = '_'.join([id_, ra, dec, f"{score:.01f}", var, channels])
    if outdir:
        if not exists(outdir):
            makedirs(outdir)
        savename = join(outdir, savename)
    import matplotlib.pyplot as plt 
    plt.switch_backend('Agg')
    transform1 = SqrtStretch() + PercentileInterval(99.8)
    transform2 = SqrtStretch() + PercentileInterval(99.95)
    transform3 = SqrtStretch() + PercentileInterval(98)
    transform4 = SqrtStretch() + PercentileInterval(99)
    im1       = fits.getdata(varpath)
    im2       = fits.getdata(normpath)
    im3       = fits.getdata(bigvarpath)
    im4       = fits.getdata(origpath)
    cmap      = plt.get_cmap('cividis')
    fig = plt.figure(figsize=(7*2.5,7))
    gs  = fig.add_gridspec(2,5)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:2, 1:3])
    ax4 = fig.add_subplot(gs[:2, 3:])
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    # plt.subplot(projection=wcs)
    ax1.imshow(transform1(im1), origin='lower', cmap=cmap)
    ax2.imshow(transform2(im2), origin='lower', cmap=cmap)
    ax3.imshow(transform3(im3), origin='lower', cmap=cmap)
    ax4.imshow(transform4(im4), origin='lower', cmap=cmap)
    for ax in (ax3, ax4):
        ax.grid(color='white', ls='solid')
    plt.suptitle(f'score:  {score:.1f}', y=0.998)
    plt.tight_layout()
    ram = io.BytesIO()
    plt.savefig(ram, format='png')
    ram.seek(0)
    im = Image.open(ram)
    im2 = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
    im2.save( savename , format='PNG')
#%%
for i in range(8000,10000):#len(data)):
    ra, dec = RA[i], DEC[i]
    offset = 5866
    name = f"{i+offset:0>5}_{ra:.04f}_{dec:.04f}"
    specworkdir = join(workdir, name)
    if not exists(specworkdir):
        mkdir(specworkdir)
    logger.info(f"{' '+ name + ' ':#^63}")
    try:
        res = downloadAndProcess(RA=ra,
                                 DEC=dec,
                                 hsize=512,
                                 workdir=specworkdir,
                                 name=name,
                                 channels=['r', 'i'],
                                 kernel_size=7,
                                 object_extent=20,
                                 skipdownload=0,
                                 redodiffimg=1,
                                 debug=0,
                                 config=config,
                                 removecontaminants=0)
        _, (bigvar, bignorm)   = res.saveVariabilityImages(maxchi2=1.5, maxseeing=5.3, crop=250, extra='big_')
        _, (varpath, normpath) = res.saveVariabilityImages(maxchi2=1.5, maxseeing=5.3, crop='same')
        print(bigvar, bignorm, varpath, normpath)
        origpath = bigvar.replace('.fits', '_originals_stack.fits')
        score = decide(normpath, debug=1, logger=logger, redo_sextractor=False)
        
        outdir_images = '/home/frederic/Desktop/diffpanstarrsoutput2'
        saveVariabilityMap(varpath, normpath, bignorm, origpath,  \
                           score, outdir=outdir_images)
        if np.isnan(score):
            logger.info('not enough information')
        elif score == -1:
            logger.info('No significant variability detected')
        detection = score > 0.7 
        final = f"{name:<40} --> {detection}   ({score:.02f})"
        logger.info(final)
        deleteFiles(specworkdir)
    except Exception as e:
        logger.error(f'error: {e}')
        logger.error(traceback.format_exc())
        print(traceback.format_exc())
    logger.info('')
    
