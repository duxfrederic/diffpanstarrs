#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:11:25 2020

@author: fred
"""

from diffpanstarrs import downloadAndProcess

# feed the routine all the parameters like the coordinates, the size of the
# images to download, which color channels, etc.
res = downloadAndProcess(RA=357.4169,
                         DEC=54.0433,
                         hsize=512,
                         workdir='myworkdir2',
                         name="unnamed2",
                         channels=['r','i'],
                         skipdownload=1)
# plot the light curves:

res.saveWeightedMedianImage(crop='same')
res.saveVariabilityImages(crop='same')
varscore, medianscore = res.score()


res.saveWeightedMedianImage(crop=500)
res.saveVariabilityImages(crop=500)
res.plotCurves()
# plot the difference images and variability image:
# res.plotDiffImg(crop=30)