#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:48:44 2020

@author: frederic
"""

import os
os.chdir('/mnt/FE867C3B867BF28F')
from diffpanstarrs import downloadAndProcess
result = downloadAndProcess (
    RA = 69.5619 ,
    DEC = -12.28745 ,
    workdir = 'toast' ,
    channels = 'r',
    skipdownload=1,
    redodiffimg=0
)
# plot the light curves :
# result.plotCurves()
# make variability maps :
result.saveVariabilityImages()
# decide whether it is a lens
# ( score between 0 and 1 ) :
print('score')
print(result.score())