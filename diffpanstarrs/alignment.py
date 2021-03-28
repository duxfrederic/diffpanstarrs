#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:07:32 2020

@author: frederic
"""
from     subprocess                 import  call
from     glob                       import  glob
from     os.path                    import  join, basename, exists
import   numpy                      as      np
from     scipy.optimize             import  minimize
from     scipy                      import  ndimage
from     astropy.io                 import  fits


from     diffpanstarrs.wcs          import  sky2pix
from     diffpanstarrs.utilities    import  clip

class field:
    """
        not immediately useful, keeping it in case we need to align
        the cutouts.
    """
    def __init__(self, RA, DEC, datadir, outdir, verbose):
        self.RA      = RA
        self.DEC     = DEC
        self.outdir  = outdir
        self.datadir = datadir
        self.verbose = verbose
 
    def initimage(self, infile, hsize):
        self.hsize   = hsize
        image        = fits.open(infile)
        x1, y1       = sky2pix(image[0].header, self.RA, self.DEC)
        pixels       = image[0].data

        XMIN         = np.max([0,int(x1)-hsize])
        XMAX         = np.min([pixels.shape[1], int(x1)+hsize])

        YMIN         = np.max([0,int(y1)-hsize])
        YMAX         = np.min([pixels.shape[0], int(y1)+hsize])

        data         = pixels[YMIN:YMAX, XMIN:XMAX].copy()
        self.data    = data
        try:
            seeing       = image[0].header['SEEING']/0.21 # 0.21 arcsecond per pixel
            zpt          = image[0].header['MAGZP']
            gain         = image[0].header['GAIN'] #this is in electrons per ADU, we have shot noise on the electrons
            self.seeing  = seeing
            self.zpt     = zpt
            self.gain    = gain
        except:
            seeing       = image[0].header['CHIP.SEEING'] # pixels, whateva
            gain         = image[0].header['CELL.GAIN']
            self.seeing  = seeing
            self.gain    = gain
        # background and noise maps
        bg, std            = clip(data, 5)
        electronmap        = abs((data-bg)*gain)
        poissonelectronmap = electronmap**0.5
        adupoissonmap      = poissonelectronmap/gain
        noisemap           = (adupoissonmap**2. + std**2.)**0.5
        self.noisemap      = noisemap
        # populate the header
        RA, DEC       = self.RA, self.DEC
        hdr           = image[0].header
        hdr['CRPIX1'] = x1-XMIN
        hdr['CRPIX2'] = y1-YMIN
        hdr['CRVAL1'] = RA
        hdr['CRVAL2'] = DEC
        primary_hdu   = fits.PrimaryHDU(header=hdr)
        hdu           = fits.PrimaryHDU(data)
        hdul          = fits.HDUList([primary_hdu, hdu])
        # write the "cropped" version
        outfile_crop = f"{infile.split('.fits')[0]}_RA_{RA}_DEC_{DEC}.fits"
        hdul.writeto(join(self.outdir, 'Sextraction', basename(outfile_crop)), \
                     overwrite=True)

    def run_sextractor(self, RA, DEC, infile, datadir, outdir, verbose):
        extrworkdir   = join(outdir, 'Sextraction')
        extractedfile = join(extrworkdir, \
                         f"{basename(infile.split('.fits')[0])}_RA_{RA}_DEC_{DEC}_hsize_{self.hsize}.cat")
        segfile       = extractedfile.split('.cat')[0] + '_seg.fits'
        sourcefile    = join(extrworkdir, \
                         f"{basename(infile.split('.fits')[0])}_RA_{RA}_DEC_{DEC}.fits")
        if not exists(extractedfile):
            call(['sex', sourcefile, '-CATALOG_NAME', extractedfile, 
                  '-CHECKIMAGE_TYPE', 'SEGMENTATION', '-CHECKIMAGE_NAME', segfile])
            print(f'attempted sextractor on file {sourcefile}')
        sexpositions = np.genfromtxt(extractedfile)
        return sexpositions[:, 1:3].T, (sexpositions[:, 3:].T)**0.5, sexpositions[:, 0].T
    


def crossmatch(positions, uncertainties, positions2, uncertainties2):
    """
        not immediately useful, keeping it in case we need to align
        the cutouts.
    """
    keep    = []
    matched = []
    for i in range(positions.shape[1]):
        x         =   positions[0][i]
        y         =   positions[1][i]
        distances = ((positions2[0]-x)**2 + (positions2[1]-y)**2.)**0.5
        if len(np.where(distances<2)[0])>0:
            keep.append(i)
            matched.append(np.argmin(distances))
    return keep, matched

def mapPixels(x,y, params):
    """
        not immediately useful, keeping it in case we need to align 
        the cutouts.
        
            len(params) == 2:    Constant offset (dx, dy)
            len(params) == 6:    linearly varying offset (dx, dy, dx(x), dx(y), dy(x), dy(y))
            len(params) == 12:   2nd order varying offset:
                                   dx,     dy,     dx(x),  dx(y),  dy(x),  dy(y), 
                                   dx(x²), dx(xy), dx(y²), dy(x²), dy(xy), dy(y²)
    """
    if not len(params) in [2, 6, 12]:
        raise AssertionError("not a valid set of parameters")
    # copy to avoid foreseeable problems:
    x, y             = x.copy(), y.copy()
    x_scale, y_scale = x / np.max(x), y / np.max(y)
    # the 0th order transformation is always applied:
    dx, dy = params[:2]
    x     += dx
    y     += dy
    
    if len(params) > 2:
        # then add the linear terms:
        dx_x, dx_y, dy_x, dy_y = params[2:6]
        x += x_scale * dx_x + y_scale * dx_y
        y += x_scale * dy_x + y_scale * dy_y
    if len(params) > 6:
        # then add the quadratic terms:
        dx_x2, dx_xy, dx_y2, dy_x2, dy_xy, dy_y2 = params[6:]
        x += x_scale**2 * dx_x2 + x_scale*y_scale * dx_xy + y_scale**2 * dx_y2
        y += x_scale**2 * dy_x2 + x_scale*y_scale * dy_xy + y_scale**2 * dy_y2
    
    return x, y


def alignmentscore(params, x1, y1, dx1, dy1, x2, y2, dx2, dy2):
    """
        not immediately useful, keeping it in case we need to align
        the cutouts.
    """
    x1p, y1p   = mapPixels(x1, y1, params)
    squares    = (x1p-x2)**2 + (y1p-y2)**2
    tot_uncert = np.sqrt(dx1**2 + dy1**2 + dx2**2 + dy2**2)
    return np.sum(squares / tot_uncert)


def fit_offset(pos1, dpos1, pos2, dpos2, mode="linear"):
    """
        not immediately useful, keeping it in case we need to align
        the cutouts.
    x1 is close to x2, size(x1) == size(x2).
    """
    x1, y1    = pos1
    dx1, dy1  = dpos1
    x2, y2    = pos2
    dx2, dy2  = dpos2
    args      = x1, y1, dx1, dy1, x2, y2, dx2, dy2
    
    if   mode == "quadratic":
        opt      = fit_offset(pos1, dpos1, pos2, dpos2, mode="linear")
        params0  = np.append(opt.x, [0, 0, 0, 0, 0, 0])
    elif mode == "linear":
        opt      = fit_offset(pos1, dpos1, pos2, dpos2, mode="zero")
        params0  = np.append(opt.x, [0, 0, 0, 0])
    else:
        params0  = [0, 0]
    
    opt = minimize(alignmentscore, x0=params0, args=args)
    
    return opt


def interpolate(image, move_coordinates_params):
    """
        not immediately useful, keeping it in case we need to align
        the cutouts.
    """
    x, y       = np.arange(image.shape[0]), np.arange(image.shape[1])
    newx, newy = mapPixels(x.astype(np.float64), y.astype(np.float64), move_coordinates_params)
    X, Y       = np.meshgrid(newx, newy)
    image      = ndimage.map_coordinates(image, [Y, X], prefilter=True)
    return image


def alignFrames(RA, DEC, datadir, outdir, verbose, hsize, seeinglimit, savereference=False):
    """
        not immediately useful, keeping it in case we need to align
        the cutouts.
    """
    filenames = sorted(glob(join(datadir, '*fits')))
    seeings   = [fits.open(filename)[0].header['SEEING'] for filename in filenames]
    seeingarg = np.argsort(seeings)

    frame     = field(RA, DEC, datadir, outdir, verbose)
    # choose a reference frame with the best seeing
    infile    = filenames[seeingarg[0]]
    frame.initimage(infile=infile, hsize=hsize)
    positions, uncertainties, fluxes = frame.run_sextractor(RA, DEC, infile, datadir, outdir, verbose)
    alpha                = np.where((fluxes>np.median(fluxes))&(fluxes<np.percentile(fluxes, 99.5)))[0]
    frame.positions      = positions.T[alpha].T
    frame.uncertainties  = uncertainties.T[alpha].T
    positions            = positions.T[alpha].T
    uncertainties        = uncertainties.T[alpha].T

    # now we register the other frames to stack to create the reference frame
    
    for aaa in range(np.where(np.array(seeings)<seeinglimit)[0].shape[0]-1):
        frame2           = field(RA, DEC, datadir, outdir, verbose)
        infile2          = filenames[seeingarg[aaa+1]]
        print(f"#################### Aligning file {basename(infile2)} ####################")
        interpoutfile    = join(outdir, 'registered', 'interpolated', \
                f"{basename(infile2.split('.fits')[0])}_RA_{RA}_DEC_{DEC}_hsize_{hsize}.fits")
        if not exists(interpoutfile):
            frame2.initimage(infile=infile2, hsize=hsize)
            positions2, uncertainties2, fluxes2 = frame2.run_sextractor(RA, DEC, \
                                                    infile2, datadir, outdir, verbose)
            alpha2               = np.where((fluxes2>np.median(fluxes2))&\
                                            (fluxes2<np.percentile(fluxes2, 99.9)))[0]
            frame2.positions     = positions2.T[alpha2].T
            frame2.uncertainties = uncertainties2.T[alpha2].T
            positions2p          = positions2.T[alpha2].T
            uncertainties2p      = uncertainties2.T[alpha2].T
            
            # act in a different order if '2' has more stars.
            # (only useful to be faster since the longer operation is vectorized)
            if positions2.size > positions.size:
                keep, matched    = crossmatch(positions, uncertainties, positions2p, uncertainties2p)
            else:
                matched, keep    = crossmatch(positions2p, uncertainties2p, positions, uncertainties)
                
            positionsT, uncertaintiesT, positions2T, uncertainties2T = \
                         positions.T[keep].T, uncertainties.T[keep].T, \
                         positions2p.T[matched].T, uncertainties2p.T[matched].T

            # find the optimal (semi local) coordinate offset to
            # match the images using a Nelder-Mead optimizer
            try:
                opt = fit_offset(positionsT, uncertaintiesT, positions2T, \
                                 uncertainties2T, mode='quadratic')
                # interpolate the image onto the first one using the 
                # transformation found just before.
                interpolatedframe = interpolate(frame2.data, opt.x)
                
                hdu = fits.PrimaryHDU(interpolatedframe)
                hdu.writeto(interpoutfile, overwrite=True)
            except:
                print(f"file {interpoutfile} to exclude")
                continue
            
            
    infile = filenames[seeingarg[0]]
    interpoutfile    = join(outdir, 'registered', 'interpolated', \
                         f"{basename(infile.split('.fits')[0])}_RA_{RA}_DEC_{DEC}_hsize_{hsize}.fits")
    if not exists(interpoutfile):
        hdu = fits.PrimaryHDU(frame.data)
        hdu.writeto(interpoutfile, overwrite=True)
    
    if savereference:
    # now stack the images
        stack = frame.data.copy()
        allframes = [stack]
        for aaa in range(np.where(np.array(seeings)<seeinglimit)[0].shape[0]-1):
            infile2 = infile  = filenames[seeingarg[aaa+1]]
            interp_infile     = join(outdir, 'registered', 'interpolated', \
                                 f"{basename(infile.split('.fits')[0])}_RA_{RA}_DEC_{DEC}_hsize_{hsize}.fits")
            if exists(interp_infile):
                interpolatedframe = fits.open(interp_infile)[1].data
                stack += interpolatedframe
                allframes.append(interpolatedframe)
    
        medianstack = np.nanmedian(np.array(allframes), axis=0)
        hdu = fits.PrimaryHDU(medianstack)
        hdu.writeto(join(outdir, 'registered', f'reference_RA_{RA}_DEC_{DEC}_hsize_{hsize}.fits'), overwrite=True)
