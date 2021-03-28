#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:14:50 2020

@author: frederic
"""

import sys
sys.path.append("..")

from   glob                   import  glob
from   os.path                import  join, basename
from   diffpanstarrs.decider  import  decide

"""
test_data = join("test_data", "test_decider")
files     = glob(join(test_data, '*fits'))
objects   = list(set([basename(e).split('_')[0] for e in files]))

for obj in objects:
    imagefile = join(test_data, obj)+'_VariabilityMap_r.fits'
    normfile  = join(test_data, obj)+'_VariabilityMap_r_normalized.fits'
    
    score = decide(normfile, redo_sextractor=False)
    
    detection = True if score > 0.7 else False 
    print(f"{obj:<28} --> {detection}   ({score:.02f})")
    print()
"""

goods = ['00304', '00305', '00357']
test_data =  "/media/frederic/01CF3B90E9BF6E00/panstarrstrials2"

cases = glob(join(test_data, '0*'))
cases.sort()
tosend = 0
threshold = 0.4

for case in cases:
    # if not '00022' in case:
        # continue
    num = basename(case).split('_')[0]
    num = int(num)
    if not (num >= 600 and num <= 1400):
        continue
    outdir = join(case, 'output')
    try:
        normimage = glob(join(outdir, '*VariabilityMap_ri_normalized.fits'))[0]
        bigoriginals = glob(join(outdir, 'big_*_VariabilityMap_ri_originals_stack.fits'))[0]
    except IndexError:
        print(f"{basename(case):<28} --> not generated")
        continue
    score = decide(normimage, bigoriginals, debug=True, redo_sextractor=True)
    print(basename(case))
    if score > threshold:
        tosend += 1
    with open('scores2', 'a') as f:
        f.write(f"{basename(case)}  \t  {score:.02f}\n")
    
    # detection = True if score > 0.1 else False 
    # print(f"{basename(case):<28} --> {detection}   {score:.02f}")
    # print()
print('to send:', tosend)
