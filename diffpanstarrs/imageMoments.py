#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:13:01 2020

@author: frederic
"""
from    os.path                   import   join, dirname, basename, exists
from    pathlib                   import   Path
from    subprocess                import   call
import  numpy                     as       np
from    astropy.io                import   fits
from    scipy.ndimage.filters     import   gaussian_filter

from    diffpanstarrs.config      import   config
from    diffpanstarrs.utilities   import   dumpDefaultSextractorFiles, \
                                           openCatFile, clip
def getGrid(shape):
    """
       simply returns a cartesian coordinate system for an image.
       Did this as to define it always in the same way.
       Unit: arcseconds (thus the 0.25" per pixel)
    """
    return 0.25 * (np.indices(shape) - (np.array(shape)//2)[:, np.newaxis, np.newaxis])


def imageCentroid(image):
    """
      yields the center of gravity of the image in the coordinates defined in
      getGrid.
      useful to compute central moments below.
    """
    X, Y = getGrid(image.shape)
    partition         = np.nansum(image)
    
    return np.nansum( X * image  ) / partition, \
           np.nansum( Y * image  ) / partition
    

def imageCentralMoment(image, p, q=None):
    """
         computes the p^{th} (x direction), q^{th} (y direction) central moment
         of the image.
         default: q = p
    """
    q = p if q is None else q
    x0, y0            = imageCentroid(image)
    X,  Y             = getGrid(image.shape)
    Xred, Yred        = X - x0, Y - y0
    
    # instead of image / noise, just mask the noise  (see ~/Desktop/diffimg3)
    # poisson           = np.sqrt(np.abs(image))
    # noise             = sigma + poisson
    partition         = np.nansum(image)
    return np.nansum( Xred**p * Yred**q  * image ) / partition


def imageSkewness(image):
    sigma_power_3 = np.abs(imageCentralMoment(image, 2))**1.5
    return np.abs(imageCentralMoment(image, 3))**(1/3) / sigma_power_3

def imageExcessKurtosis(image):
    """
        simply computes the normalized 4th central moment of the image, 
        and removes 3 from the result.
    """
    sigma_power_4 = np.abs(imageCentralMoment(image, 2))
    return np.abs(imageCentralMoment(image, 4))**0.5  / sigma_power_4 - 3

def imageEllipticity(image):
    Mxx   = imageCentralMoment(image, 2, 0)
    Myy   = imageCentralMoment(image, 0, 2)
    Mxy   = imageCentralMoment(image, 1, 1) 
    Mxxyy = Mxx + Myy
    Me1   = (Mxx - Myy) / Mxxyy
    Me2   = 2 * Mxy / Mxxyy
    val   = (Mxx - Myy)**2 + Mxy**2
    return val, Me1, Me2


def imageFindPatches(diffimg, insane_blur=1, redo=1):
    """
    to be used on crops of the variability maps obtained by a stack
    of difference images. 
    
    Parameters
    ----------
    diffimg : string
        path to the variability map to be analyzed. preferably a crop 
        of the region of interest.
    insane_blur : bool
        if true, completely blur the map before running sextractor as to
        detect only one source. 
    redo :    bool
        re-run sextractor even if the catalog is already present?
    
    

    Returns
    -------
    numpy array 1D, 
        flux of each detection.
    tuple 
        tuple contaning the x and y positions (numpy arrays)
    tuple
        tuple containing the semi major and minor axes length of each detection.
        (numpy arrays)
    numpy array 2D
        segmentation map of the detections
    """
    
    dumpDefaultSextractorFiles()
    directory = dirname(diffimg)
    catfile   = join(directory, basename(diffimg).replace('.fits', '.cat'))
    segfile   = join(directory, basename(diffimg).replace('.fits', '_seg.fits'))
    if insane_blur:
        img    = fits.open(diffimg)[0]
        header = img.header
        size   = 1
        img    = img.data
        bc, st = clip(img, 5)
        img[np.isnan(img)] = bc
        img    = gaussian_filter(img, size, mode='constant', cval=bc)
        hdu    = fits.PrimaryHDU(img)
        hdu.header = header
        hdu.header['HISTORY'] = f"blurred with a gaussian kernel of radius {size}"
        diffimg = Path(str(diffimg).replace('.fits', '_blurred.fits'))
        if not diffimg.exists():
            redo = 1
        hdu.writeto(str(diffimg), overwrite=1)
        
        
    if (not exists(segfile) or not exists(catfile)) or redo:
        options = [config.sextractor, str(diffimg), '-DETECT_THRESH', '1.5', #'-DEBLEND_MINCONT', '1',
                   '-CATALOG_NAME', str(catfile),
                   '-CHECKIMAGE_TYPE', 'SEGMENTATION', '-CHECKIMAGE_NAME', segfile]
        call(options)
    isoflux, (x,y), (a,b) = openCatFile(catfile)
    return isoflux, (x,y), (a,b), fits.open(segfile)[0].data
    

if __name__ == "__main__":
    # """#### check for a gaussian
    size = 30
    x, y = np.indices((size, size)) 
    X, Y = x - size // 2, y - size // 2
    sigma = 2
    x0, y0 = 0, 0
    Z0  =  np.exp( -1/2 * ( (X-1)**2 + (Y)**2 ) / sigma**2 )
    Z0 += 0.1* np.exp( -1/2 * ( (X+7)**2 + (Y)**2 ) / sigma**2 )
    # Z0[10:20, size//4] = 0.5
    # Z0 = np.exp(- sigma * np.abs(X) - sigma * np.abs(Y))
    import matplotlib.pyplot as plt
    # Z0 = moffatProfile(2, 3, 1, 0, 0, 1, 1 , 0, 0, (size,size))
    plt.clf()
    plt.imshow(Z0)
    # print(np.abs(imageCentralMoment(Z0, 3))**0.25)
    # print(imageExcessKurtosis(Z0))
    # print(imageSkewness((Z0)))
    print(imageEllipticity(Z0))
    ##########################################"""
    """#### test on a single diff img
    directory = "/home/frederic/Desktop/panstarrstrials/SDSS153559.97+430819.0/data"
    from glob import glob
    images = glob(join(directory, "*.i.*.fits"))
    #base      = "rings.v3.skycell.2201.073.wrp.i.55340_35292.fits"
    images = [image for image in images if (not '.mask.' in image) \
                                       and (not '.wt.'   in image)]
    for image in images:
        base  = basename(image)
        path  = join(directory, base)
        dpath = join(directory, 'output', base.replace('.fits', '_diff.fits'))
        wpath = join(directory, 'output', base.replace('.fits', '_diff.wt.fits'))
        spath = join(directory, 'output', 'Sextraction', base.replace('.fits', '_seg.fits'))
        import matplotlib.pyplot as plt
        try:
            a  = fits.open(path)[0]
            da = fits.open(dpath)[0]
            wa = fits.open(wpath)[0]
            sa = fits.open(spath)[0]
        except:
            print('problem with base', base)
            plt.close()
            continue
        # x0, y0 =  510, 514
        listofstars = [(901,777), (52,745), (542, 570), (97,498), (677,422), (420,309), \
                       (464,304), (982,204), (380,182), (521, 503), (510, 514)]
        star1_quasar2 = [(521, 503), (510, 514)]
        print(base)
        n = 8
        fig, axstot = plt.subplots(2,3)
        for (x0, y0), axs in zip(star1_quasar2, axstot) :
            dreg = da.data[y0-n:y0+n, x0-n:x0+n]
            reg = a.data[y0-n:y0+n, x0-n:x0+n]
            seg = sa.data[y0-n:y0+n, x0-n:x0+n]
            
            ax1, ax2, ax3 = axs
            dregtest = dreg.copy()
            if np.sum(np.isnan(dregtest)) > 0.1 * dregtest.size:
                break
            # dregtest[seg==0] = np.nan
            ax1.imshow(reg, origin='lower')
            ax2.imshow(dregtest, origin='lower')
            ax3.imshow(seg, origin='lower')
            val = imageExcessKurtosis(dregtest)
            ax2.set_title(f'excess curtosis: {val:.02f}')
            print((x0, y0), '-->', val)
        plt.tight_layout()
        plt.waitforbuttonpress()
        plt.close('all')
        print()
    ##########################################"""
    """ test on a variability map
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_erosion
    paths = ["/home/frederic/Desktop/panstarrstrials/HE0435-1223/data/output/VariabilityMap_r.fits",
             "/home/frederic/Desktop/panstarrstrials/SDSS153559.97+430819.0/data/output/VariabilityMap_r.fits",
             "/home/frederic/Desktop/panstarrstrials/SDSSJ1051+5922A/data/output/VariabilityMap_r.fits"]
    for path in paths:
        image = fits.open(path)[0].data
        (isoflux, (x,y), (a,b)), segmap = imageFindPatches(path)
        n = 6
        for i in range(1, x.size + 1):
            xi, yi = int(x[i-1]), int(y[i-1])
            segmaps = segmap[yi-n:yi+n, xi-n:xi+n]
            images = image[yi-n:yi+n, xi-n:xi+n]
            mask = (segmaps == i)
            # mask = binary_erosion(mask, iterations=1)
            imgtest = images.copy()
            imgtest[~mask] = np.nan
            plt.figure()
            plt.imshow(imgtest, origin='lower')
            val = imageExcessKurtosis(imgtest)
            plt.title(f'excess kurtosis: {val:.02f}')
            print("###########################################")
            print(f'({xi:.0f}, {yi:.0f}) ->', val)
        plt.waitforbuttonpress()
        input('press enter to continue ')
        plt.close('all')
        
        
    #"""
    
