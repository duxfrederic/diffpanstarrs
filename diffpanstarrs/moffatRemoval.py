#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:20:39 2020

@author: frederic
"""
import   numpy               as       np
from     scipy.optimize      import   minimize

from     diffpanstarrs.imageMoments  import  imageCentroid, getGrid

def moffatProfile(alpha, beta, scaling, x0, y0, a, b, theta, background, shape):
    """
    Parameters
    ----------
    alpha : float
        alpha parameter of a Moffat profile.
    beta : float
        beta parameter of a Moffat profile.
    scaling: float
        scaling of the Moffat profile.
    x0 and y0: floats
        correction to the center.
    a and b:   floats
        semi major and semi minor axes of the ellipse
    shape : tuple
        shape of the square grid on which the Moffat profile is centered.

    Returns
    -------
    2D numpy array
        Moffat profile sampled and centered on the returned array.
    """
    ygrid, xgrid = getGrid(shape)
    theta = -theta
    xgrid = xgrid - x0
    ygrid = ygrid + y0
    xgrid = (np.cos(theta) * xgrid - np.sin(theta)*ygrid)
    ygrid = (np.sin(theta) * xgrid + np.cos(theta)*ygrid)
    prefactor    = (beta - 1) / (np.pi * alpha**2)
    factor       = (1 + ( (xgrid)**2 / a**2 + (ygrid)**2 / b**2 )  / alpha**2)**(-beta)
    return (scaling * prefactor) * factor + background
    
def produceGridForImageCrop(shape):
    """
        make x and y coordinates centered in the center of the image, where
        the center of the contaminating star is supposed to be found.
    """
    return np.indices(shape) - (np.array(shape)//2)[:, np.newaxis, np.newaxis]

def moffatScore(params, *args):
    """
    Parameters
    ----------
    params : array
        alpha and beta parameters of a Moffat profile, as well as a scaling.
    *args : tuple
        data and noise images for the fit in the sense of minimal chi2.

    Returns
    -------
    float
        sum of the squared residuals.

    """
    imagecrop, dimagecrop, mask = args
    shape                 = imagecrop.shape
    residuals             = moffatProfile(*params, shape) - imagecrop
    residuals             = residuals[mask]
    dimagecrop            = dimagecrop[mask]
    return np.nansum( (residuals / dimagecrop)**2 )


def fitMoffatProfile(imagecrop, dimagecrop, mask):
    """
    Fit a Moffat Profile in the sense of the minimization of the χ² to the
    image imagecrop with uncertainties dimagecrop.

    Parameters
    ----------
    imagecrop : 2d numpy array
        data image.
    dimagecrop : 2d numpy array
        uncertainties, pixelwise (noise image).

    Returns
    -------
    tuple
        alpha and beta parameters.
    χ2 : float
        reduced chi-squared of the least-squares fit.

    """
    alphaini   = 5
    betaini    = 1.8
    scalingini = 50 * np.nanpercentile(imagecrop, 99)
    x0, y0     = imageCentroid(imagecrop)
    a, b       = 1, 1
    theta      = 0.
    x0         = [alphaini, betaini, scalingini, x0, y0, a, b, theta, 0]
    opt = minimize(moffatScore, x0=x0, 
                                args=(imagecrop, dimagecrop, mask),
                                tol=1e-8)
    chi2     = opt.fun / (np.sum(~np.isnan(imagecrop[mask])) - len(x0))
    
    return (opt.x, chi2)


def removeMoffatProfile(image, imagestd, segmap, lim, xcont, ycont, id_star_cont):
    lim = int(lim)
    import matplotlib.pyplot as plt
    vmin, vmax = np.nanpercentile(image, [0.5, 99.5])
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    for x, y, id_star in zip(xcont, ycont, id_star_cont):
        x, y = int(x), int(y)
        test = image[y-lim:y+lim, x-lim:x+lim].copy()
        test2 = segmap[y-lim:y+lim, x-lim:x+lim]
        test[test2!=segmap[y,x]] = 0
        # plt.imshow(test, vmin=vmin, vmax=vmax, origin='lower')
        params, chi2 = fitMoffatProfile(image[y-lim:y+lim, x-lim:x+lim], 
                                        imagestd[y-lim:y+lim, x-lim:x+lim],
                                        segmap[y-lim:y+lim, x-lim:x+lim]==segmap[y,x])
        background = params[-1]
        print(f"      removing a moffat profile on the star at ({x},{y}) with a reduced chi2 of {chi2:.02f}")
        lim = int(1.4*lim)
        ax1.imshow(image[y-lim:y+lim, x-lim:x+lim].copy(), vmin=vmin, vmax=vmax, origin='lower')
        model = moffatProfile(*params, (2*lim, 2*lim))
        image[y-lim:y+lim, x-lim:x+lim] -=  model - background
    
    
    ax2.imshow(model, vmin=vmin, vmax=vmax, origin='lower')
    ax3.imshow(image[y-lim:y+lim, x-lim:x+lim], vmin=vmin, vmax=vmax, origin='lower')
    plt.show(block=True)
    plt.close()


if __name__ == "__main__":
    #%%
    path = "/home/frederic/Desktop/panstarrstrials/SDSS153559.97+430819.0/data/rings.v3.skycell.2201.073.wrp.i.55340_35292.fits"
    spath = "/home/frederic/Desktop/panstarrstrials/SDSS153559.97+430819.0/data/output/Sextraction/rings.v3.skycell.2201.073.wrp.i.55340_35292_seg.fits"
    import matplotlib.pyplot as plt
    from astropy.io import fits
    a = fits.open(path)[0]
    da = fits.open(path.replace('.fits', '.wt.fits'))[0]
    sa = fits.open(spath)[0]
    y0, x0 =  1003, 879
    n = 10
    im = a.data[x0-n:x0+n, y0-n:y0+n]
    dim = np.sqrt(da.data[x0-n:x0+n, y0-n:y0+n])
    sim = sa.data[x0-n:x0+n, y0-n:y0+n]
    mask = sim == 5
    params, chi2 = fitMoffatProfile(im, dim, mask)
    # params = [1.8, 1.1, 3.27e4, -2, 1, 1, 1.0, -0.2]
    model = moffatProfile(*params, im.shape)
    # plt.clf()
    # plt.plot(model[:, 10])
    # plt.errorbar(np.arange(20), im[:,10], yerr=dim[:,10])
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(im)
    ax2.imshow(model)
    # ax2.imshow(a.data[x0-n:x0+n, y0-n:y0+n] - model, origin='lower')
    # plt.imshow(model)
    plt.title(f'chi2: {chi2:.01f}')
