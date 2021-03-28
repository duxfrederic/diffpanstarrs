#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:25:02 2020

@author: frederic
"""
from pathlib import Path
from astropy.io import fits
import numpy as np


cat_file = 'gaia_doubles_w1w2_0.5_RA_0to360_G_16p5_3as_W1_16.5_dec_m30_b15_0p8_to_3as_pmsig_5_wdensities.fits'
cat_path = Path('test_data') / 'news_catalogue' / cat_file
data = fits.getdata(cat_path)
def selectRowsFromCatalogue(data):
    print(f"Initial table:                 {len(data)} entries.")
    
    # select only very high redshifts:
    dW = 0.7
    data = data[np.where(data['WISE_W1'] - data['WISE_W2']> dW)]
    print(f"After excluding low redshifts: {len(data)} entries left.")
    
    # select only those with a low proper motion significance:
    maxpmsig = 3
    data = data[ np.where(   (data['MAX_PMSIG'] < maxpmsig) * ~np.isnan(data['MAX_PMSIG'])  ) ]
    print(f"After excluding high motion:   {len(data)} entries left.")
    
    # now sort by magnitude:
    data.sort(order='gaia1_phot_g_mean_mag')
    
    return data
    
data = selectRowsFromCatalogue(data)