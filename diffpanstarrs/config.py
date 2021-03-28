#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:51:15 2020

@author: fred
"""
import os
from   shutil import  which
from   .      import  sextractor_defaults

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

class Config():
    def __repr__(self):
        msg = "############## extra parameters for diffpanstarrs: ##############\n"
        msg += f"kernel_size:             \t {self.kernel_size}\n"
        msg += f"sextractor:              \t {self.sextractor}\n"
        msg += f"min_number_stars:        \t {self.min_number_stars}\n"
        msg += f"magnitude_threshold_star:\t {self.magnitude_threshold_star}\n"
        msg += "################################################################"
        return msg
    def __str__(self):
        return self.__repr__()

    download_target_url = "http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={RA}&dec={DEC}&type=warp"
    download_outdir = 'downloaded_panstarrs'
    sextractor = which('sex') or which('sextractor') or which('source-extractor')
    if not sextractor:
        raise AssertionError("sextractor not found in the path, give config.py the exact location of the sextractor binaries")
    min_number_stars = 20
    crop_border_pixels = 1
    magnitude_threshold_star = 16
    kernel_size = 7
    object_extent = 20
    debug = False
    channels = ['g', 'i', 'r', 'y', 'z']
    defaultparam = pkg_resources.read_text(sextractor_defaults, 'default.param')
    defaultconv  = pkg_resources.read_text(sextractor_defaults, 'default.conv')
    nprocessesdownload = min(os.cpu_count(), 10)

config = Config()
