#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 22:14:10 2020

@author: frederic
"""

from glob import glob
from os.path import join
import numpy as np

from diffpanstarrs import decide

directory = "/media/frederic/kingston_data/varmaps"
#%%
ys = []
for  d in glob(join(directory, '*')):
    try:
        varfits = glob(join(d, 'output', '*VariabilityMap_ri_normalized.fits'))[0]
        ys.append(decide(varfits))
    except IndexError:
        pass
    
ys = np.array(ys)
ys[ys>0.5] = 1
ys[ys<=0.5] = 0

print(len(ys), 'amid which', ys.sum(), 'are candidates according to the CNN')