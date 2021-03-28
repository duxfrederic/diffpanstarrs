#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 08:59:18 2019

@author: frederic dux
"""

import  urllib3
import  wget
from    time                       import  time
from    os.path                    import  join, exists, dirname
from    shutil                     import  move 
from    os                         import  makedirs, remove
from    multiprocessing            import  Pool
from    subprocess                 import  call

from    diffpanstarrs.utilities    import  binFitsFile, dumpDefaultSextractorFiles
from    diffpanstarrs.config       import  config


def unpackLine(line, headers):
    entry   = {header:element for header, element in zip(headers, line.split(' '))}
    return entry


def parseRequest(url):
    """
        reads a list of file locations on the panstarrs public server and formats
        them as a python list.
    """
    http    = urllib3.PoolManager()
    r       = http.request('GET', url, preload_content=0)
    
    data    = r.data.decode('utf-8')
    data    = data.split('\n')
    headers = data[0].split(' ')
    data    = data[1:]
    return [unpackLine(line, headers) for line in data if line]

def actual_download(entry):
    """
    will be passed to the threads in downlaodData. 
    
    entry contains the download parameters.

    """
    (outimaget, basetarget, Nimage, keep_original, hsize, binning), entry = entry
    print(f"  Downloading images for epoch {entry['shortname']}")
    outimage   = outimaget.format(filename=entry['shortname'])
    urlimage   = basetarget.format(filename=entry['filename'],\
                                   RA=entry['ra'],
                                   DEC=entry['dec'],
                                   width=hsize,
                                   shortname=entry['shortname']
                                   )
    # also generate the filenames and urls for the auxiliary data:
    outmask    = outimage.replace('.fits', '.mask.fits')
    outweight  = outimage.replace('.fits', '.wt.fits')

    urlmask    = urlimage.replace('.fits', '.mask.fits')
    urlweight  = urlimage.replace('.fits', '.wt.fits')
    
    # download and bin everything accordingly:
    try:
        for outfile, url in zip([outimage, outmask, outweight], [urlimage, urlmask, urlweight]):
            if not exists(outfile):
                wget.download(url, out=outfile)
                if binning > 1:
                    binFitsFile(outfile, 2)
                    if not keep_original:
                        remove(outfile)
    except Exception as e:
        print(f"      ----> error with epoch {entry['shortname']}: {e}")
                    
                    
def downloadData(outdir, RA, DEC, hsize=1024, binning=1, keep_original=1, url=config.download_target_url,
                 channels=config.channels):
    """
        downloads all the available panstarrs data that satisfies the requirements
        given in the parameters.
    """
    # first, check that the field has enough stars. If no, download a larger field.
    N = checkNumberOfStars(outdir, RA, DEC, hsize)
    if N == -1:
        return
    if N < config.min_number_stars:
        hsizenew = 2*hsize    
        hsize = min(hsizenew, 1024)
        print(f"Found only {N} stars in the field. hsize is increased to {hsize}.")
    # but should not go further than 1024 in size, which is already quite big.
    
    url = url.format(RA=RA, DEC=DEC)
    # redondant
    if not exists(outdir):
        makedirs(outdir)
    parsed_request = parseRequest(url)
    parsed_request_filtered  = []
    for channel in channels:
        parsed_request_filtered += [e for e in parsed_request if f".{channel}." in e['shortname']]
    parsed_request = parsed_request_filtered
    
    # preparing the urls to contact:
    outimaget      = join(outdir, "{filename}")
    
    basetarget     = "http://ps1images.stsci.edu/cgi-bin/fitscut.cgi?red={filename}"
    basetarget    += "&&format=fits&x={RA}&y={DEC}&size={width}&wcs=1&imagename={shortname}"
    
    # 3 files (wt, mask, image) for each epoch:
    Nimage         = len(parsed_request)
    print(f"######### Downloading. There are images to download for {Nimage} epochs.")
    # since we are using multirpocessing and I don't want to write 200 lines,
    # give some overhead with the global info to each download:
    t0 = time()
    to_download = [ ((outimaget, basetarget, Nimage, keep_original, hsize, binning), entry) for entry in parsed_request ]
    
    
    pool = Pool(config.nprocessesdownload)
    pool.map(actual_download, to_download)
    pool.close()
    pool.join()
    print(f"######## Download completed, took {time()-t0:.0f} seconds to complete.")


def checkNumberOfStars(outdir, ra, dec, hsize):
    if not exists(outdir):
        makedirs(outdir)
    imagename = 'i_field_verification_stars'
    service = "http://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = ("{service}?ra={ra}&dec={dec}&size={hsize}&format=fits"
           "&filters=i").format(**locals())
    http    = urllib3.PoolManager()
    try:
        file    = http.request('GET', url, preload_content=0).data.decode('utf-8').split('\n')[1].split(' ')[-3]
    except IndexError:
        print(f'No data available to download at {ra}, {dec}.')
        return -1
    url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           "ra={ra}&dec={dec}&size={hsize}&format=fits").format(**locals())
    newname = join(outdir, imagename+'.fits')
    wget.download(url+'&red='+file, out=newname)
    catfile = newname.replace('.fits', '.cat')
    dumpDefaultSextractorFiles()
    options = [config.sextractor, newname, '-CATALOG_NAME', catfile]
    call(options)
    t = sum(1 for line in open(catfile))
    return t

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Script downloading panSTARRS data")
    parser.add_argument('--RA',  type=float, help='Right ascension coordinate (float format)')
    parser.add_argument('--DEC', type=float, help='Declination coordinate (float format)')
    parser.add_argument('--hsize', type=int, help='Desired size of the field (pixels)', default=1024)
    parser.add_argument('--binning', type=int, help='Should the images be binned?', default=1)
    parser.add_argument('--outdir', type=str, help='Directory where to download the files', default=config.download_outdir)
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

