#!/usr/bin/env python

from diffpanstarrs.plotting import plotThisDirectory
import  argparse

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