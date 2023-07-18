#!/usr/bin/env python
import argparse
from    diffpanstarrs.plotting         import  plotThisDirectory
from    diffpanstarrs.getPanSTARRSData import  downloadData
from    diffpanstarrs.config           import  config

def plot_this_directory():
    parser = argparse.ArgumentParser(description="Interactive plot of the images in a given directory")
    parser.add_argument('path', type=str, help='Path to the directory', default='')
    parser.add_argument('--crop', type=int, default=0)
    parser.add_argument('--pattern', type=str, default='')
    parser.add_argument('--sum', type=int, default=False)
    parser.add_argument('--datetime', type=int, default=0, help="Can a datetime be extracted from the filename?")
    parser.add_argument('--removenan', type=int, default=0, help="Skip the files with nans in the center?")
    parser.add_argument('--globalnormalize', type=bool, default=False, help="Normalize the white point and black point with respect to the whole data set?")
    args = parser.parse_args()
    print(args)
    plotThisDirectory(args.path, pattern=args.pattern, 
                                 crop=args.crop, 
                                 removenan=args.removenan,
                                 absolutesum=args.sum, 
                                 datetime=args.datetime, 
                                 headless=False,
                                 globalnormalize=args.globalnormalize)


def download_panstarrs_data():
    parser = argparse.ArgumentParser(description="Script downloading panSTARRS data")
    parser.add_argument('--RA',  type=float, help='Right ascension coordinate (float format)')
    parser.add_argument('--DEC', type=float, help='Declination coordinate (float format)')
    parser.add_argument('--hsize', type=int, help='Desired size of the field (pixels)', default=1024)
    parser.add_argument('--binning', type=int, help='Should the images be binned?', default=1)
    parser.add_argument('--outdir', type=str, help='Directory where to download the files',
                        default=config.download_outdir)
    args = parser.parse_args()

    RA      = args.RA
    DEC     = args.DEC
    hsize   = args.hsize
    outdir  = args.outdir
    binning = args.binning
    if not RA or not DEC:
        print("No coordinates were given :(")
        import sys
        sys.exit()
    downloadData(outdir, RA, DEC, hsize, binning)
