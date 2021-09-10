#!/usr/bin/env python
from os.path import join
from os import environ
import argparse 
import sys
sys.path.append("..")

from diffpanstarrs.pipeline import downloadAndProcess

def test_simple_download_and_process():
    workdir = join(environ['HOME'], "test_diffpanstarrs_simple_download_and_process")
    
    
    parser = argparse.ArgumentParser(description='test the pipeline on this simple example')
    
    parser.add_argument('-RA', type=float, default=69.5619)
    parser.add_argument('-DEC', type=float, default=-12.28745)
    parser.add_argument('-hsize', type=int, default=1024)
    parser.add_argument('-channels', nargs='+', type=str, default='r')
    parser.add_argument('-skipdownload', type=bool, default=False)
    parser.add_argument('-debug', type=bool, default=False)
    parser.add_argument('-workdir', type=str, default=workdir)
    parser.add_argument('-name', type=str, default='HE0435-1223')
    parser.add_argument('-kernel_size', type=int, default=9)
    parser.add_argument('-redo', type=bool, default=True)
    args = parser.parse_args()
    
    res = downloadAndProcess(args.RA, args.DEC, args.hsize,
                             workdir=args.workdir, 
                             name=args.name,
                             channels=args.channels,
                             kernel_size=args.kernel_size,
                             skipdownload=args.skipdownload,
                             debug=args.debug,
                             redodiffimg=args.redo)
    
    
    
    a = input('plot? [y,N] ')
    if a == 'y': 
        res.plotCurves()
        res.plotDiffImg(crop=30)

if __name__ == "__main__":
    test_simple_download_and_process()