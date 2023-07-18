#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:59:36 2019

@author: frederic
"""
from    os.path           import      join, exists, basename, dirname
from    os                import      makedirs, stat, get_terminal_size
from    sys               import      argv
from    subprocess        import      call
from    glob              import      glob
import  numpy             as          np
from    astropy.io        import      fits
from    shutil            import      move


from    diffpanstarrs.differenceImaging import      differenceImaging
from    diffpanstarrs.getPanSTARRSData  import      downloadData
from    diffpanstarrs.plotting          import      plotLightCurves, plotThisDirectory
from    diffpanstarrs.utilities         import      saveIntensities, areThereNansInTheCenter,\
                                                    makeAbsoluteStackOfDiffImg,\
                                                    recallChi2AndSeeing, loadPanSTARRSauxiliary,\
                                                    dumpDefaultSextractorFiles, chooseBestImgPerNight,\
                                                    recallDiffImgs, weightedMedianImage
from    diffpanstarrs.config            import      config

class DiffImgResult():
    """
         a container to manipulate the result of the routine.
         The accessible properties are:

             RA:             the RA coordinate
             DEC:            the DEC coordinate
             hsize:          the size of the images that were downloaded and analyzed
             name:           the custom name of the analyzed field
             workdir:        the directory where the raw data was downloaded
             outdir:         the directory where the output images and curves were
                             stored
             channels:       the color channels that were downloaded and analyzed
             kernel_size:    the size of the kernel used in the difference imaging
                             procedure.
             object_extent:  the width of the box in the center in which
                             the intensity of the difference images
                             is summed.

         The accessible methods are (see docstrings for details):
             plotCurves:
                         yields a plot of the computed light curves
             plotDiffImg:
                         yields a plot of the computed difference images
             plotOriginals:
                         yields a plot of the downloaded data.

    """
    def __init__(self, RA, DEC, hsize, name, workdir, channels, kernel_size):
        inits = locals()
        self.__dict__.update(inits)
        del self.__dict__["self"]
        self.outdir  = join(workdir, 'output')
        self.light_curve_paths = {}
        self.varimages = {}
        self.varimagesnorm = {}
        self.varimagesnormfiles = {}
        self.medianimages = {}
        self.medianimagesfiles = {}
        self.object_extent = None
        
    def __repr__(self):
        try:
            width = get_terminal_size().columns
        except:
            width = 60
        filler = "#"
        messg = width * filler + '\n'
        messg += " Difference imaging result ".center(width, filler) + '\n'
        messg += f" field {self.name} at RA={self.RA:.03f}, DEC={self.DEC:.03f} ".center(width,filler)+ '\n'
        messg += width * filler + '\n'
        messg += f"  output files at: {join(self.workdir, self.outdir)}" + '\n'
        return messg
    
    def __str__(self):
        return self.__repr__()

    def plotCurves(self):
        plotLightCurves(self.light_curve_paths, self.name)
        
    def saveVariabilityImages(self, channels=None, maxchi2=1.5, maxseeing=7, crop=None, extra=''):
        if crop == "same":
            crop = self.object_extent
        if not channels:
            channels = self.channels
        fitspath = join(self.outdir, self.name +'_VariabilityMap_{channel}.fits')
        files = []
        for channel in channels:
            if not channel in self.channels:
                print(f"  {channel} not available in this analysis")
                continue
            files  += recallDiffImgs(self.outdir, channel)
        if len(files) == 0:
            raise Exception("No image available to build a variability map.")
        chi2s, seeings = zip(*[recallChi2AndSeeing(self.outdir, file) for file in files])
        chi2s, seeings = np.array(chi2s), np.array(seeings)
        # here, implement an heuristic that chooses the best map every night.
        try:
            filesg = chooseBestImgPerNight(files)
            print(f"     saving a variability map using {len(filesg)} out of {len(files)} files.")
        except FileNotFoundError:
            print("      No file with the right criteria found to build the varmap.")
            return "", ""
        
        channel = ''.join(channels)
        outfits = fitspath.format(channel=channel)
        outfits = join(dirname(outfits), extra+basename(outfits))
        varimage, varpath = makeAbsoluteStackOfDiffImg(filesg, outfits=outfits,\
                                              crop=crop, normalize=False)
        varnorm, normpath = makeAbsoluteStackOfDiffImg(filesg, outfits=outfits,\
                                              crop=crop, normalize=True)
        self.varimages[channel] = varimage
        self.varimagesnorm[channel] = varnorm
        self.varimagesnormfiles[channel] = normpath
        return varpath, normpath
    
    def saveWeightedMedianImage(self, channels=None, crop=None, extra=''):
        if crop == "same":
            crop = self.object_extent
        if not channels:
            channels = self.channels 
        fitspath = join(self.outdir, self.name +'_WeightedMedianImage_{channel}.fits')
        files = []
        for channel in channels:
            files += recallDiffImgs(self.outdir, channel)
        channel = ''.join(channels)
        outfits = fitspath.format(channel=channel)
        outfits = join(dirname(outfits), extra+basename(outfits))
        median, medianfits = weightedMedianImage(files, crop, outfits)
        self.medianimagesfiles[channel] = medianfits
        self.medianimages[channel] = median
        return medianfits
        

    def plotDiffImg(self, channels=None, crop=False, removenan=False, savename='',
                    headless=False):
        """
            plots the requested channels of the difference images.

            Arguments:
                channels : list or string
                    list of strings denoting the channels or
                    a single channel.
                    e.g. channels = ['r', 'i']  or channels = 'i'
                    DEFAULT: all of them
                crop : int
                    if given, crops to the center with a width of
                   'crop' pixels.
                removenan : int
                    if given, excludes all the images with
                    bad pixels in the center in a width of
                    'removenan' pixels.
        """
        if not channels:
            channels = self.channels
        self._plotImg(channels, crop, removenan, directory=self.outdir,\
                      absolutesum=True, savename=savename, headless=headless)

    def plotOriginals(self, channels, crop=False, removenan=False, savename='',
                      headless=False):
        """
            plots the requested channels of the downloaded data.

            Arguments:
                channels : list or string
                    list of strings denoting the channels or
                    a single channel.
                    e.g. channels = ['r', 'i']  or channels = 'i'
                    DEFAULT: all of them
                crop : int
                    if given, crops to the center with a width of
                   'crop' pixels.
                removenan : int
                    if given, excludes all the images with
                    bad pixels in the center in a width of
                    'removenan' pixels.
        """
        if not channels:
             channels = self.channels
        self._plotImg(channels, crop, removenan, directory=self.workdir,\
                      absolutesum=False, savename=savename, headless=headless)

    def _plotImg(self, channels, crop, removenan, directory, absolutesum,
                 savename, headless):
        """
            plots all the img in a directory.
        """
        if not type(channels) is list:
            channels = [channels]
        for channel in channels:
            if not channel in self.channels:
                print(f"  {channel} not available in this analysis")
                continue
            plotThisDirectory(directory, pattern=f".{channel}.", crop=crop,
                              removenan=removenan, absolutesum=absolutesum,
                              savename=savename, headless=headless)



def extractRA_DEC_name(dirpath):
    actualname = basename(dirpath)
    RA, DEC, hsize, name = actualname.split('_')
    RA, DEC, hsize = float(RA), float(DEC), int(hsize.replace('hsize', ''))
    return RA, DEC, hsize, name


def excludeSeries(file, reason='corrupt'):
    base = file.replace('.wt.fits','').replace('.mask.fits','').replace('.fits','')
    for ext in ['.wt.fits', '.mask.fits', '.fits']:
        if exists(base+ext):
            move(base + ext, base + ext + "." + reason)


def excludeCorrupt(dirpath):
    fitsfiles = glob(join(dirpath, '*.fits'))
    for file in fitsfiles:
        try:
            _ = fits.open(file)[0].data
        except Exception as e:
            print(e)
            excludeSeries(file, reason='corrupt')

def excludeMissing(dirpath):
    fitsfiles = glob(join(dirpath, '*fits'))
    bases = [file.replace('.wt.fits','').replace('.mask.fits','').replace('.fits','') for file in fitsfiles]
    bases = list(set(bases))
    for base in bases:
        for ext in ['.wt.fits', '.mask.fits', '.fits']:
            if not exists(base+ext):
                excludeSeries(base, reason='missing')


def doDifferenceImagingPanSTARRS(sourcedir, name, kernel_size, object_extent,
                                 result, redo=False, channels=config.channels, debug=config.debug,
                                 removecontaminants=False, config=config):
    # make sure the output director exists
    outdir      = join(sourcedir, 'output')
    sexout      = join(outdir, 'Sextraction')

    if not exists(outdir):
        makedirs(outdir)
    if not exists(sexout):
        makedirs(sexout)

    for channel in channels:
        # in this file, we will save the light curve of this channel:
        tosave_csv = join(outdir, f'{name}_{channel}.txt')
        if len(argv)<2 and exists(tosave_csv):
            # if we give the script an argument (whichever), we force recompute
            # everything.
            if not stat(tosave_csv).st_size == 0 and not redo:
                print(f'{name} (channel {channel}) already processed')
                # in this case, proceed to the next channel
                result.light_curve_paths[channel] = tosave_csv
                continue

        # what is below this happens if we decide that we proceed
        images      = sorted(glob(join(sourcedir, f'*.{channel}.*.fits')))
        images      = [image for image in images if not '.mask.' in image and not '.wt.' in image]
        imagesfiltered = []
        for image in images:
            try:
                # apparently, there are some corrupt files and astropy.io should
                # complain here, thus filtering the image
                # at the same time, remove the useless images (that contain nan bands
                # in their center)
                flag = areThereNansInTheCenter(fits.open(image)[0].data, npixels=1)
            except:
                flag = True
                print(f"file {image} seems to be corrupt.")
                move(image, image+'.corrupt')
                noisemap = image.replace('.fits', '.wt.fits')
                mask     = image.replace('.fits', '.mask.fits')
                move(noisemap, noisemap+'.corrupt')
                move(mask, mask+'.corrupt')
            if not flag:
                imagesfiltered.append(image)
        images = imagesfiltered
        # now sextraction:
        dumpDefaultSextractorFiles()
        imageswithenoughstars = []
        # save the paths to the catalogs and segmentation maps while we're at it:
        catsegfiles = {}
        for image in images:
            catfile = join(sexout, basename(image).replace('.fits', '.cat'))
            segfile = join(sexout, basename(image).replace('.fits', '_seg.fits'))
            catsegfiles[image] = catfile, segfile
            if not exists(segfile) or not exists(catfile):
                options = [config.sextractor, image, '-CATALOG_NAME', catfile,
                               '-CHECKIMAGE_TYPE', 'SEGMENTATION', '-CHECKIMAGE_NAME', segfile]
                call(options)
                if sum(1 for line in open(catfile)) < config.min_number_stars:
                    excludeSeries(image, reason='notenoughstars')
                else:
                    imageswithenoughstars.append(image)
            else:
                if sum(1 for line in open(catfile)) >= config.min_number_stars:
                    imageswithenoughstars.append(image)

        # we are left with the images containing enough stars:
        images      = imageswithenoughstars
        if len(images) < 2:
            print(f"Not enough images in the {channel} band. Stopping.")
            continue
        # extract the seeings from the headers:
        seeings     = np.array([ fits.open(image, memmap=1)[0].header['CHIP.SEEING']  for image in images ])
        # now, we start preparing the difference imaging:
        # the reference array is built from a combination of as few images as possible
        seeings     = [fits.open(image, memmap=1)[0].header['CHIP.SEEING'] for image in images]
        # say as a rule of thumb that whatever happens, we want at least 70% of our
        # images to have a less good seeing than the reference array.
        reffiles    = [image for image in images if fits.open(image)[0].header['CHIP.SEEING'] < np.percentile(seeings, 50)]
        # sort the files by seeing
        reffiles.sort(key=lambda reffile: fits.open(reffile)[0].header['CHIP.SEEING'])
        # and put them together until at least the center is well 
        # covered. 
        j = 0
        refarray    = fits.getdata(reffiles[0])
        firstloop   = True
        refarraystack, refarraywtstack = [], []
        while areThereNansInTheCenter(refarray, object_extent*3) or firstloop:
            # else we open more and more images and combine them:
            im         = fits.getdata(reffiles[j])
            mask, dim  = loadPanSTARRSauxiliary(reffiles[j], mask_outset=0)
            im[~mask]  = np.nan
            dim[~mask] = np.nan
            refarraystack.append(im)
            refarraywtstack.append(dim)
            
            partition     = np.nansum(1/np.array(refarraywtstack), axis=0) 
            nperpixel     = np.sum(~np.isnan(refarraystack), axis=0)
            refarray      = np.nansum(np.array(refarraystack)/np.array(refarraywtstack), axis=0) / partition
            refarraywt    = nperpixel / partition
            firstloop = False
            j += 1
            # this is when we give up.
            if j > len(reffiles)-1:
                break
            print(j)
            
        intensities = {}
        

        # difference image each image in the set with respect to the reference:
        for image in images:
            try:
                difftot, tot, chi2 = differenceImaging(infile=image,
                                                       refarray=refarray,
                                                       refarraystd=refarraywt,
                                                       outdir=outdir,
                                                       magnitude_threshold=config.magnitude_threshold_star,
                                                       kernel_size=kernel_size,
                                                       object_extent=object_extent,
                                                       redo=redo,
                                                       plotintermediate=debug)

                intensities[image] =  difftot, tot, chi2
                
            except Exception as e:
                import traceback
                print('\033[91m')
                print(traceback.format_exc())
                print(e)
                print('\033[0m')
                intensities[image] = np.nan, np.nan, np.nan
                
        # book keeping:
        saveIntensities(intensities, saved_csv_name=tosave_csv)
        result.light_curve_paths[channel] = tosave_csv


def downloadAndProcess(RA, DEC, hsize=512, workdir=None, name="unnamedregion",
                       channels=None,
                       kernel_size=None,
                       object_extent=None, \
                       redodiffimg=False, skipdownload=False,
                       debug=None, removecontaminants=False, config=config):
    """
       The pipeline that downloads and  processes a single region of the sky identified by its
       coordinates and size (hsize).

       Arguments:
         RA : float
                 right ascension coordinate in degrees.
         DEC : float
                 declination coordinate in degrees.
         hsize : int
                 size in pixels of the square images to download and process.
         workdir : string
                  directory where the data will be downloaded and processed.
                  DEFAULT: current_work_dir/RA_DEC_name
                  where name is the next parameter.
         name : string
                  A name for the region being analyzed.
                  DEFAULT: unnamedregion
         channels : list
                  A list of color channel. The default is all the
                  Pan-STARRS channel as defined in diffpanstarrs.config.channels
                  DEFAULT: diffpanstarrs.config.channels, namely ['g', 'i', 'r', 'y', 'z']
         kernel_size : int
                  the size of the kernel used in the difference imaging procedure.
         object_extent : int
                  the width of the box in the center in which
                  the intensity of the difference images
                  is summed.
         redodiffimg : bool
                  toggles the reprocessing of the difference images. Note that the light
                  curves are always reprocessed by the mean of loading the computed
                  difference images into the memory, even if redodiffimg is False.
                  DEFAULT: False
         skipdownload : bool
                  entirely skips the download part, useful to avoid loosing time
                  on checking whether all the files are here.
       Returns:
           DiffImgResult object
    """
    if not channels:
        channels = config.channels
    if not kernel_size:
        kernel_size = config.kernel_size
    if not object_extent:
        object_extent = config.object_extent
    if not debug:
        debug = config.debug
    if not workdir:
        workdir = f"{RA}_{DEC}_{name}"
    if not skipdownload:
        downloadData(workdir, RA, DEC, hsize, channels=channels)
    excludeCorrupt(workdir)
    excludeMissing(workdir)
    try:
        width = get_terminal_size().columns
    except:
        width = 60
    print(width*"#")
    print(f" Processing field in {basename(workdir)} ".center(width, "#"))
    print(width*"#"+"\n")
    result               = DiffImgResult(RA, DEC, hsize, name, workdir, channels, kernel_size)
    result.config        = config
    result.object_extent = object_extent
    
    doDifferenceImagingPanSTARRS(sourcedir=workdir,
                                 name=name,
                                 kernel_size=kernel_size,
                                 object_extent=object_extent,
                                 redo=redodiffimg,
                                 channels=channels,
                                 result=result,
                                 debug=debug,
                                 removecontaminants=removecontaminants,
                                 config=config)

    return result
