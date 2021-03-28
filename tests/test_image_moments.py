#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:58:17 2020

@author: frederic
"""

import sys
sys.path.append("..")

from   pathlib            import Path
from   astropy.io         import fits
import matplotlib.pyplot  as     plt; plt.switch_backend('TKAgg')
from   scipy.ndimage      import binary_erosion

from diffpanstarrs.imageMoments import imageFindPatches, imageExcessKurtosis


# def test_image_moments():
datadir = Path('test_data') / 'test_image_moments'
for path in [p for p in datadir.glob('*Variability*.fits') if not ('_seg' in p.stem or 'blurred' in p.stem)]:
    print(f' analyzing image {path.name} '.center(60, '#'))
    image = fits.open(path)[0].data
    isoflux, (x,y), (a,b), segmap = imageFindPatches(path, insane_blur=False, redo=False)
    n = 6
    for i in range(1, x.size + 1):
        xi, yi = int(x[i-1]), int(y[i-1])
        segmaps = segmap[yi-n:yi+n, xi-n:xi+n]
        images = image[yi-n:yi+n, xi-n:xi+n]
        mask = (segmaps == i)
        mask = binary_erosion(mask, iterations=1)
        imgtest = images.copy()
        # imgtest[~mask] = 0.001
        plt.figure()
        plt.imshow(imgtest, origin='lower')
        val = imageExcessKurtosis(imgtest)
        plt.title(f'excess kurtosis: {val:.02f}')
        print(f'({xi:.0f}, {yi:.0f}) ->', val)
        plt.waitforbuttonpress()
    plt.close('all')

# if __name__ == "__main__":
#     test_image_moments()