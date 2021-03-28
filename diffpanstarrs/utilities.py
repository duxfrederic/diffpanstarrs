#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 19:16:36 2019

@author: frederic dux
"""

from    os.path                import  basename, exists, join
from    datetime               import  datetime
import  csv
import  numpy                  as      np
from    scipy.ndimage          import  binary_erosion, binary_dilation
from    astropy.io             import  fits
from    astropy.nddata.utils   import  Cutout2D
from    astropy                import  wcs as astropywcs
import  robustats

from    diffpanstarrs.config   import  config



def weightedMedianImage(listOfFiles, crop, outfits):
    try:
        stack, noiseMaps, updatedwcs = loadStackOfDiffImg(listOfFiles, crop, \
                                                          outfits, normalize=False)
    except ValueError:
        if exists(outfits):
            result = fits.getdata(outfits)
            return result, outfits
        else:
            print("Cannot proceed without data or pre existing image")
            raise ValueError 
    stack = np.abs(stack)
    ny, nx = stack[0].shape 
    result = np.zeros((ny, nx))
    for n in range(nx*ny):
        row, col = n//nx, n%nx
        valuecolumn = stack[:, row, col]
        weightcolumn = noiseMaps[:, row, col]
        where = np.where(~np.isnan(valuecolumn))
        valuecolumn = valuecolumn[where]
        weightcolumn = weightcolumn[where]
        if valuecolumn.size == 0:
            result[row, col] = np.nan 
        else:
            result[row, col] = robustats.weighted_median(valuecolumn, weightcolumn)
    if outfits:
        hdu = fits.PrimaryHDU(result)
        hdu.header = fits.open(listOfFiles[-1])[0].header 
        hdu.header.update(updatedwcs.to_header())
        hdu.writeto(outfits, overwrite=True)
    return result, outfits


def chooseBestImgPerNight(files: list):
    times = [fits.open(file)[0].header['MJD-OBS'] for file in files]
    nights = {}
    for file, time in zip(files, times):
        roundedtime = int(time+0.5)
        if not roundedtime in nights:
            nights[roundedtime] = [file]
        else:
            nights[roundedtime].append(file)
    chosenfiles = []
    for night, nightfiles in nights.items():
        chi2s   = [fits.open(file)[0].header['REDCHI2'] for file in nightfiles]
        seeings = [fits.open(file)[0].header['CHIP.SEEING'] for file in nightfiles]
        sorting = np.argsort(seeings)
        seeings, chi2s, nightfiles = np.array(seeings)[sorting], np.array(chi2s)[sorting], np.array(nightfiles)[sorting]
        for seeing, chi2, file in zip(seeings, chi2s, nightfiles):
            if chi2 < 1.5 and seeing < 6 and seeing > 4:
                chosenfiles.append(file)
                break 
        else:
            chosenfiles.append(nightfiles[np.argmin(chi2)])
    return chosenfiles
        
            
def dumpDefaultSextractorFiles():
    if not exists('default.param'):
        with open('default.param', 'w') as f:
            f.writelines(config.defaultparam)
        print(" ~~~~  Created a default.param file for sextractor ~~~~ ")
    if not exists('default.conv'):
        with open('default.conv', 'w') as f:
            f.writelines(config.defaultconv)
        print(" ~~~~  Created a default.conv file for sextractor  ~~~~ ")
        


def loadPanSTARRSauxiliary(image_filename, mask_outset=-2):
    """
        given the base filename NAME.fits of an image,
        loads the noise map (NAME.wt.fits) and mask (NAME.mask.fits)
        and does some cleaning.
        
            image_filename:     the main image path: "path/NAME.fits"
            
            mask_outset:        errosion or dilatation of the mask, number of pixels
                                to add to the borders.
    """
    mask_path   = image_filename.replace('.fits', '.mask.fits')
    weight_path = image_filename.replace('.fits', '.wt.fits')
    
    mask   = fits.open(mask_path)[0].data
    weight = fits.open(weight_path)[0].data
    
    mask[~np.isnan(mask)] = 0
    mask[np.isnan(mask)]  = 1
    mask = mask.astype(np.bool)
    if mask_outset > 0:
        mask = binary_erosion(mask, iterations=mask_outset)
    elif mask_outset < 0:
        mask = binary_dilation(mask, iterations=-mask_outset)
    
    # the (inverse) weight maps are actually variance maps. We need the std.
    std  = np.sqrt(weight) 
    
    return mask, std
    


def extractDateTime(filename):
    """
        extracts a datetime object from a filename. not useful for panstarrs.
    """
    FORMAT = '%Y-%m-%d'
    extracted = basename(filename).split('_')[1]
    return datetime.strptime(extracted, FORMAT)


def saveIntensities(intensities_dic, saved_csv_name):
    w = csv.writer(open(saved_csv_name, "w")) 
    
    for image, (dintensity, intensity, chi2) in intensities_dic.items():
        time = fits.open(image)[0].header['MJD-OBS']
        w.writerow([basename(image), time, dintensity, intensity, chi2])


def getMagnitudes(catfile, offset):
    sexpositions =  np.genfromtxt(catfile)
    mags         = -2.5 * np.log10(sexpositions[:, 0].T)
    return mags + offset


def openCatFile(catfile):
    data    = np.genfromtxt(catfile)
    if len(data.shape) == 1:
        data = data.reshape(1, data.shape[0])
    if data.size < 1:
        return [], ([], []), ([], [])
    isoflux = data[:,0].T
    x, y    = data[:,1].T, data[:, 2].T
    a, b    = data[:,3].T, data[:, 4].T
    return isoflux, (x,y), (a,b)


def findCloserThanSeeing(catfile, maxseeing, hsize, n=3):
    _, (x,y), _ = openCatFile(catfile)
    x0  = y0    = hsize//2 + 1
    indices     = np.arange(1, x.size + 1)
    distances   = np.sqrt( (x-x0)**2 + (y-y0)**2 )
    bad         = np.where(distances < n * maxseeing)
    indices     = indices[bad]
    distances   = distances[bad]
    x, y        = x[bad], y[bad]
    sorting     = np.argsort(distances)
    return x[sorting][1:], y[sorting][1:], distances[sorting][1:], indices[sorting][1:]
        
    
        
def cropToCenter(array2D, npixels, wcs=None):
    centerx       = array2D.shape[0]//2
    centery       = array2D.shape[1]//2
    array2Dcrop   = array2D[centerx-npixels:centerx+npixels,centery-npixels:centery+npixels]
    array2Dcrop   = Cutout2D(array2D, (centerx, centery), 2*npixels, wcs=wcs)
    if wcs:
        return array2Dcrop
    else:
        return array2Dcrop.data


def areThereNansInTheCenter(array2D, npixels=10):
    if np.max(np.isnan(cropToCenter(array2D, npixels=npixels))):
        return True
    elif -9223372036854775808 in cropToCenter(array2D, npixels=npixels):
        # this is the int value of a nan. never know
        return True
    else:
        return False
    
    
def binArray(array2d, ratio):
    assert ratio in [2**n for n in range(10)]
    ncol, nrow = array2d.shape
    shapenew   = ncol//ratio, nrow//ratio
    arraytmp   = array2d[:shapenew[0]*ratio, :shapenew[1]*ratio]
    shapetmp   = (shapenew[0], ratio, shapenew[1], ratio)
    return arraytmp.reshape(shapetmp).mean(axis=3).mean(axis=1)


def binFitsFile(path, ratio, outpath=''):
    struct = fits.open(path)[0]
    struct.header['CRPIX1'] /= ratio
    struct.header['CRPIX2'] /= ratio
    struct.header['CDELT1'] *= ratio
    struct.header['CDELT2'] *= ratio
    struct.header['BAXIS1']  = ratio
    struct.header['BAXIS2']  = ratio
    struct.data = binArray(struct.data, ratio)
    if not outpath:
        outpath = path.replace('.fits', f'_binned{ratio}x{ratio}.fits')
    fits.writeto(outpath, struct.data, struct.header, overwrite=1)
    

def loadDataArrayFromListOfFiles(listOfFiles, datetime=True):
    if datetime:
        return {extractDateTime(f) : fits.open(f)[1].data for f in listOfFiles}
    else:
        try:
            return {basename(f) : fits.open(f)[1].data for f in listOfFiles}
        except:
            return {basename(f) : fits.open(f)[0].data for f in listOfFiles}
    

def makeAverageOriginalImages(listOfFiles, crop, outfits):
    originals = [fits.open(f)[0].header['srcfile'] for f in listOfFiles]
    noisesf   = [f.replace('.fits', '.wt.fits') for f in originals]
    
    images, noises = [], []
    w = astropywcs.WCS(originals[0])
    for diff, im, noise in zip(listOfFiles, originals, noisesf):
        diff  = fits.open(diff)[0].data 
        im    = fits.open(im)[0].data
        noise = fits.open(noise)[0].data**0.5

        if crop:
            diff  = cropToCenter(diff, crop)
            im    = cropToCenter(im, crop)
            noise = cropToCenter(noise, crop)
        else:
            im    = cropToCenter(im, diff.shape[0]//2)
            noise = cropToCenter(noise, diff.shape[0]//2)
        im[np.isnan(diff)]    = np.nan
        noise[np.isnan(im)]   = np.nan
        images.append(im)
        noises.append(noise)
    images, noises = np.array(images), np.array(noises)
    
    N_per_pixel    = np.sum(~np.isnan(images))
    
    partition      = np.nansum(1/noises, axis=0)
    average        = np.nansum(np.abs(images) / noises, axis=0) / partition  / N_per_pixel
    # we will also need the noise map of this average:
    # it will be the sum in quadrature of all the noise values that contributed.
    noisemap       = N_per_pixel / partition
    # now update the wcs if we cropped. Probably not the pythonic way but
    # will do for now; better than do it everytime in the loop.
    if crop:
        updatedwcs = cropToCenter(fits.open(listOfFiles[0])[0].data, crop, wcs=w).wcs
    else:
        updatedwcs = w
    
    hdu        = fits.PrimaryHDU(average)
    hdu.header = fits.open(listOfFiles[0])[0].header
    hdu.header.update(updatedwcs.to_header())
    hdu.writeto(outfits.replace('.fits', '_originals_stack.fits'), overwrite=True)
    hdunoise   = fits.PrimaryHDU(noisemap**2)
    hdunoise.header = hdu.header 
    hdunoise.writeto(outfits.replace('.fits', '_originals_stack.wt.fits'), overwrite=True)
    return average, noisemap




def loadStackOfDiffImg(listOfFiles, crop, outfits, normalize):
    if len(listOfFiles) == 0:
        print('no files fed here')
        raise Exception("no files to stack in makeAbsoluteStackOfDiffImg")
    listOfArrays = []
    noiseMaps    = []
    try:
        for f in listOfFiles:
            noise = f.replace('.fits', '.wt.fits')
            listOfArrays.append(fits.open(f)[0].data)
            noiseMaps.append(fits.open(noise)[0].data**0.5)
    except FileNotFoundError:
        print('the data was probably deleted.')
        if normalize:
            outfits = outfits.replace('.fits', '_normalized.fits')
        if exists(outfits):
            print('  but a former variability map was found')
            return fits.getdata(outfits), outfits
        else:
            return None, None
    w = astropywcs.WCS(f)
    
    if crop:
        updatedwcs   = cropToCenter(listOfArrays[0], crop, wcs=w).wcs
        listOfArrays = [cropToCenter(a, crop) for a in listOfArrays]
        noiseMaps    = [cropToCenter(n, crop) for n in noiseMaps]
    else:
        updatedwcs   = w
    listOfArrays, noiseMaps = np.array(listOfArrays), np.array(noiseMaps)
    # the noise maps are not originally masked:
    noiseMaps[np.isnan(listOfArrays)] = np.nan
    noiseMaps[noiseMaps==0]           = np.nan
    
    return listOfArrays, noiseMaps, updatedwcs



def makeAbsoluteStackOfDiffImg(listOfFiles, crop=False, outfits=False, 
                               normalize=False):
    try:
        listOfArrays, noiseMaps, updatedwcs = loadStackOfDiffImg(listOfFiles, crop, \
                                                                 outfits, normalize)
    except ValueError:
        return loadStackOfDiffImg(listOfFiles, crop, \
                                  outfits, normalize)
    # the weights going into the variability image:
    N_per_pixel = np.sum(~np.isnan(listOfArrays), axis=0)

    partition   = np.nansum(1/noiseMaps, axis=0)
    # make the variability image:
    varimage    = np.nansum(np.abs(listOfArrays)/noiseMaps, axis=0) / partition / N_per_pixel
    # now, we need the updated weights of the new stacked image:
    noisemap    = N_per_pixel / partition
    
    # now, we might want to have this relative relative to the intensities
    # of the original image:
    if normalize:
        originals_stack, originals_stack_noisemap = makeAverageOriginalImages(listOfFiles, crop, outfits)
        # varimagenorm  = np.abs(varimage / (originals_stack**2 + originals_stack_noisemap**2 )**0.5 )
        varimagenorm  = np.abs(varimage / originals_stack_noisemap )
        # now the variance map of the normalized variability image:
        # (introducing short names for convenience)
        ΔSD = noisemap # this is the variance map of the variability map
        ΔSI = originals_stack_noisemap # this is the variance map of the stack of originals 
        SI  = originals_stack # this is the stack of originals
        SD  = varimage # this is the variability image, Stack of Difference images
        dvarimagenorm = np.abs(np.sqrt(  np.abs( (ΔSD * (SI + ΔSI) - SD * ΔSI) / (SI + ΔSI)**2 ) ))
    if outfits:
        hdu = fits.PrimaryHDU(varimage)
        hdu.header = fits.open(listOfFiles[-1])[0].header 
        hdu.header.update(updatedwcs.to_header())
        hdu.writeto(outfits, overwrite=True)
        # the variance map now:
        hduw = fits.PrimaryHDU(noisemap**2)
        hduw.header = fits.open(listOfFiles[-1])[0].header 
        hduw.header.update(updatedwcs.to_header())
        hduw.writeto(outfits.replace('.fits', '.wt.fits'), overwrite=True)
        if normalize:
            hdu.data = varimagenorm
            hdu.header['HISTORY'] = 'normalized by the original files'
            normimagepath = outfits = outfits.replace('.fits', '_normalized.fits')
            hdu.writeto(normimagepath, overwrite=True)
            # and the variance map, again:
            hduw.data = dvarimagenorm**2
            hduw.header['HISTORY'] = 'normalized by the original files (noise map)'
            dnormimagepath = normimagepath.replace('.fits', '.wt.fits')
            hduw.writeto(dnormimagepath, overwrite=True)
    return varimage, outfits
    

def clip(data, nsigma):
    """
    estimates the background level and its standard deviation

    Parameters
    ----------
    data : numpy array
        image.
    nsigma : int
        nsigma from the background to be included.

    Returns
    -------
    tuple
        background, noise.

    """
    lennewdata = 0
    lenolddata = data.size
    while lenolddata>lennewdata:
        lenolddata = data.size
        data       = data[np.where((data<np.nanmedian(data)+nsigma*np.nanstd(data)) & \
                                   (data>np.nanmedian(data)-nsigma*np.nanstd(data)))]
        lennewdata = data.size
    return np.nanmedian(data), np.nanstd(data)


_chi2logfilename = "diff_img_chi2_seeing.txt"
def recallChi2AndSeeing(outdir, diffimgfilename):
    toread = join(outdir, _chi2logfilename )
    if not exists(toread):
        raise("Difference imaging not performed")
    with open(toread, 'r') as r:
        line = r.readline().replace('\n', '')
        chi2s = []
        while line:
            filename, chi2, seeing = line.split('\t')
            chi2s.append(float(chi2))
            if diffimgfilename:
                if basename(diffimgfilename) == basename(filename):
                    return float(chi2), float(seeing)
            line = r.readline().replace('\n', '')
    if not diffimgfilename:
        return chi2s
    else:
        return None

def recallDiffImgs(outdir, channel):
    toread = join(outdir, _chi2logfilename )
    if not exists(toread):
        raise RuntimeError("Difference imaging not performed")
    filenames = []
    with open(toread, 'r') as r:
        line = r.readline().strip()
        while line:
            filename, chi2, seeing = line.split('\t')
            if f'.{channel}.' in filename:
                filenames.append(join(outdir, basename(filename)))
            line = r.readline().strip()
    return filenames

def logChi2AndSeeing(outdir, diffimgfile, chi2, seeing):
    newline = f"{diffimgfile}\t{chi2:.02f}\t{seeing:.02f}\n"
    toopen = join(outdir, _chi2logfilename)
    lines = []
    if not exists(toopen):
        with open(toopen, 'a+') as f:
            f.write(newline)
        return 1
    with open(toopen, 'r') as r:
        for line in r.readlines():
            file, _,  _ = line.split('\t')
            if not basename(file) == basename(diffimgfile):
                lines.append(line) 
    lines.append(newline)
    with open(toopen, 'w') as f:
        for line in lines:
            f.writelines(line)
    return 0


if __name__ == "__main__":
    from glob import glob 
    datadir = "/home/fred/stuff/*diff.fits"
    files = glob(datadir)
    result = weightedMedianImage(files, False, 'toast.fits')
    
    
