#!/usr/bin/env python
"""
a module to detect and centroid sources and then perform a flux-conserving interpolation 
of the data in the same fashion as the ISIS difference imaging software

Cameron Lemon & Frederic Dux
"""
# libraries:
from     os.path                import  join, exists, basename
import   numpy                  as      np
from     scipy                  import  ndimage
import   astropy.io.fits        as      fits
import   matplotlib.pyplot      as      plt


from    diffpanstarrs.utilities     import  cropToCenter, getMagnitudes, clip,\
                                            loadPanSTARRSauxiliary, logChi2AndSeeing,\
                                            openCatFile
from    diffpanstarrs.moffatRemoval import  removeMoffatProfile


def buildStarMask(segfilepath, catfilepath, magnitude_threshold, mask_background=True,
                  object_extent=16):
    """
        Builds a star mask based on the sextracted catalogue and segmentation map.
        Excludes stars brighter than magnitude_threshold from the mask.
        Also excludes the background by default as it does not contain information
        about the PSF.
    """
    magnitudes   = getMagnitudes(catfilepath, offset=26)
    print(f"      {len(magnitudes)} stars detected. {len(magnitudes[magnitudes>=magnitude_threshold])} to be used.")
    _, _, (a,b)  = openCatFile(catfilepath)
    # load the corresponding segmap
    segmap       = fits.open(segfilepath)[0].data
    # get the id of the stars that are brighter than some magnitude threshold
    bright_ids   = np.where(magnitudes <  magnitude_threshold)
    dim_ids      = np.where(magnitudes >= magnitude_threshold) 
    # build a mask
    starmask     = np.ones_like(segmap, dtype=np.bool)
    # do not take stars that are too anisotropic (probably shouting stars)
    anisotropy_limit = 2
    # also save the stars that are too bright:
    toobrights   = np.zeros_like(segmap, dtype=np.bool)
    for bright_id, bigaxis, smallaxis in zip(bright_ids[0], a[bright_ids], b[bright_ids]):
        starmask[ segmap == bright_id+1 ] = False
        if not ( bigaxis/smallaxis > anisotropy_limit or smallaxis/bigaxis > anisotropy_limit ):
            toobrights[ segmap == bright_id + 1] = True
    for dim_id, bigaxis, smallaxis in zip(dim_ids[0], a[dim_ids], b[dim_ids]):
        if ( bigaxis/smallaxis > anisotropy_limit or smallaxis/bigaxis > anisotropy_limit):
            starmask[ segmap == dim_id+1 ] = False
    if mask_background:
        starmask[ segmap == 0 ] = False
    # add more space around the stars as to prevent larger kernels from
    # just pushing all the bad pixels outside of the mask
    starmask   = ndimage.binary_dilation(starmask,   iterations=4)
    toobrights = ndimage.binary_dilation(toobrights, iterations=1)
    # now, also mask the central object:
    nx, ny = starmask.shape
    radius = object_extent
    starmask[nx//2-radius:nx//2+radius, ny//2-radius:ny//2+radius] = False
    toobrights[nx//2-radius:nx//2+radius, ny//2-radius:ny//2+radius] = False
    return starmask, toobrights

def buildLSQMatrix(N, image, background=True):
    columns = []
    maxval        = (N + 1) // 2 - 1 # by how much can we translate the refarray
    for bb in range(N):
        for aa in range(N): # for each translation of the ref image
            column = np.roll(image, shift=(-maxval+bb,-maxval+aa), axis=(0,1))
            columns.append(column.flatten())
    # (constant) background row:
    if background:
        columns.append(np.ones(image.size))
    # we could add more rows to give more freedom to the fit of the background
    # (e.g. terms linear in the x or y coordinates) <- not done  here, did not
    # seem useful.
    fullmatrix = np.array(columns).T
    return fullmatrix

def differenceImaging(infile, refarray, refarraystd, outdir, magnitude_threshold,\
                     kernel_size, object_extent, redo=False, plotintermediate=False):
    """
    Blurs the refarray image such that its PSF matches that of the image contained in 'infile'.
    The refarray is convolved with a NxN kernel with the aim that the refarray and infile image 
    match in the sense of the weighted least squares of the sum of their pixel-wise difference.
    (χ² minimization. A successful minimization will often lead to χ^2 values << 1,
    because of the large number of degrees of freedom an NxN kernel contains.)
    
    The brightest stars and the background are masked in the process, such that
    only the useful pixels (those covering 'healthy' stars) are used in the linear
    regression.

    If working with panstarrs data, noise maps and masks are available and used
    by the algorithm to exclude more pixels. Naming convention:
        image:       image.fits
        noise maps:  image.wt.fits
        masks:       image.mask.fits
    The latter two should be found in the same directory as that of the first.


    Parameters
    ----------
    infile : str
        path to the target image onto which the reference image is to be adapted.
    refarray : 2D array image
        reference image, should be the best seeing image of the stack.
    refarraystd : 2D array, noise map
        the noise map of the reference image. Can be set to None.
    outdir : str
        path to where to difference image should be saved.
    magnitude_threshold : float
        magnitude under which the stars of the field should be masked.
    kernel_size : int
        The size of the bluring kernel. The algorithm cost scales as N². 
        odd number prefered such that a central pixel exists (no forced translation). 
        The default is 15.
    object_extent : int
        how large is the (central) object of interest? The sum over a square
        of size object_extent is taken at the end of the procedure. The default is 20.
    redo : bool, optional
        Whether to repeat the analysis if an ouptut file already exists (overwrite). 
        The default is False.
    border : int, optional
        the images are cropped by this number. The default is 1.
    plotintermediate : bool, optional
        Generates a plot of the kernel, the image, the difference image, etc. 
        Useful for debugging. The default is False.

    Returns
    -------
    float
        the intensity of the central region (20x20 pixels) of the difference image.
    TYPE
        the intensity of the central region (20x20 pixels) of the  reference image.

    """
    print(f" --> Processing file {basename(infile)}")
    targetpath    = infile
    diffoutname   = join(outdir, basename(targetpath).replace('.fits', '_diff.fits'))
    diffwtoutname = join(outdir, basename(targetpath).replace('.fits', '_diff.wt.fits'))
    if not exists(targetpath):
        print(f" --> File {targetpath} has not been generated. Skipping.")
        return np.nan, np.nan
    
    if not exists(diffoutname) or redo:
        # load the image:
        targetfits   = fits.open(targetpath)[0]
        targetheader = targetfits.header
        target       = targetfits.data#[border:-border, border:-border]
        
        # also load the masks and noise maps provided by
        # the panstarrs server.
        targetmask, targetstd = loadPanSTARRSauxiliary(targetpath, mask_outset=-2)
        targetstd_copy = targetstd.copy()
        # now, if there are stars to remove, do it here:
        ### TODO : put that elsewhere
        # if removecontaminants:
            # removeMoffatProfile(target, targetstd, targetmask, lim, *closestars)

        # now, for the following to make sense we would need to blur the
        # refarray noise map with the result. And that would be an 
        # iterative procedure since the likelihood function would be updated
        # as well.
        ## if not refarraystd is None:
            ## targetstd = np.sqrt(targetstd**2 + refarraystd**2)
        """
        # instead, we just try to rescale the noise map of the image to be processed
        # s.t. it includes the reference at the 0th order:
        refbckmed, refbckstd = clip(refarraystd[~np.isnan(refarraystd)], 5)
        tarbckmed, targckstd = clip(targetstd[~np.isnan(targetstd)], 5)
        # we just rescale everything by what's in the background in the reference
        # and the target:
        targetstd    = targetstd / (refbckstd/refbckmed) * (targckstd/tarbckmed)
        targetstd = np.sqrt(targetstd**2 + targetstd**2)
        """

            
            
        if plotintermediate:
            # useful when absurd things happen
            fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2,3, figsize=(13,11), sharex='all', sharey='all')
            cmap = plt.get_cmap('cividis')
            ax1.set_title('target image')
            ax2.set_title('reference')
            ax3.set_title('blured reference')
            ax5.set_title('selected pixels')
            ax6.set_title('transition kernel $K$')
            shax = ax6.get_shared_x_axes()
            shay = ax6.get_shared_y_axes()
            shax.remove(ax6)
            shay.remove(ax6)
            for ax in [ax1,ax2,ax3,ax4]:
                ax.set_aspect('auto')
                ax.set_xticks([]); ax.set_yticks([])
            vmin, vmax = np.nanpercentile(refarray, 1), np.nanpercentile(refarray, 99.6)
            ax1.imshow(target, vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
            ax2.imshow(refarray, vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
            # plt.show comes later, the figure isn't complete yet
    
        # start constructing the least square matrix
        refarray_copy = refarray.copy()
        refarray_copy[np.isnan(refarray_copy)] = 0
        std_copy = refarraystd.copy()
        std_copy[np.isnan(refarray_copy)] = 0
        N = kernel_size
        fullmatrix = buildLSQMatrix(N, refarray_copy)
    
        # remove the edges since they cannot be used to constrain the convolution kernel
        yy, xx  = np.indices(np.shape(refarray))
        yy, xx  = yy.flatten(), xx.flatten() 
        k1      = np.where( (yy>yy.min()+N) & (yy<yy.max()-N) & (xx>xx.min()+N) & \
                            (xx<xx.max()-N) )[0]                                     
        target2 = target.flatten()[k1]
        
        # load the star map
        segfilepath   = join(outdir, 'Sextraction', basename(infile).replace('.fits', '_seg.fits'))
        catfilepath   = join(outdir, 'Sextraction', basename(infile).replace('.fits', '.cat'))
        
        
        # save a copy, useful at the end to eliminate the bad pixels from the difference
        # image:
        notflatmask   = targetmask.copy()
        targetmask = targetmask[N+1:-N-1, N+1:-N-1].flatten()
        targetstd  = targetstd[N+1:-N-1, N+1:-N-1].flatten()
            
        # now building a star mask. brighstars2 will be kept in its original form and used
        # later to mask the very bright stars out of the difference images.
        brightstars, toobrights = buildStarMask(segfilepath, catfilepath, magnitude_threshold, 
                                                 mask_background=True, object_extent=object_extent)
        brightstars  = brightstars[N+1:-N-1, N+1:-N-1]
        # here removing the borders like we did with all the other images:
        shapeini     = brightstars.shape
        k            = brightstars.flatten()
        
        # eliminate more pixels  using the target mask and the borders:
        k *= targetmask
        k *= ndimage.binary_erosion(~np.isnan(refarray.flatten()[k1]), iterations=5)
        targetstd = targetstd[k]
            
        # now the least squares procedure. We weigh the columns of the matrix
        # and the image by the noise map.
        M       = fullmatrix[k1][k]
        target3 = target2[k]
        M       = M       / targetstd[:,np.newaxis]
        target3 = target3 / targetstd
        # last check, we do not want any NaNs to go through:
        notnan  = np.where(~np.isnan(target3))
        # least square. Sometimes the matrix is ill-conditioned, we make
        # sure to use only the bigger singular values by setting rcond=0.01
        # (only the singular values larger than 1% of the largest are used)
        x       = np.linalg.lstsq(M[notnan], target3[notnan], rcond=0.01)
        # now, the kernel is contained in x.
        # we perform the convolution to obtain the blured reference image newimg.
        newimg  = np.dot(fullmatrix, x[0]).reshape(refarray.shape) 
        # restore the bad parts that cannot be used:
        newimg[np.isnan(refarray)] = np.nan
        # and the difference image is ........the difference.
        diffimg = newimg - target
       
        # eliminate all the bad pixels and burnt stars:
        diffimg[~notflatmask] = np.nan
        diffimg[toobrights]   = np.nan
        
        
        del(fullmatrix)
        # now blur the weight map of the reference:
        # (we don't add the background this time)
        fullmatrix = buildLSQMatrix(N, std_copy**2, background=False)
        # also we suppose the pixels are normaly distributed and we
        # add the uncertainties in quadrature, weighted by the same kernel.
        newstd     = np.dot(fullmatrix, x[0][:-1]).reshape(std_copy.shape)
        newstd     = np.sqrt(newstd)
        newstd[np.isnan(refarray)] = np.nan
        newstd    /= x[0][:-1].sum()
        
        diffstd = np.sqrt(targetstd_copy**2 + newstd**2) / 2**0.5
        
        # for tracking, compute the score (count only the pixels that were
        # actually used in the minimization)
        score   = diffimg.flatten()[k1][k]/diffstd.flatten()[k1][k]
        # we will also need the number of pixels this represents:
        Npix    = (score[~np.isnan(score)]).size
        # to compute the reduced χ²
        redχ2 = np.nansum(score**2)/(Npix-N**2) 
        print(f"      Number of pixels elected for the minimization: {Npix}")
        print(f"      Number of pixels in the kernel: {N**2}")
        print(f"      {N}x{N} kernel reduced χ2 {redχ2:.2f}")
        
        # we write the combined uncertainties of the reference and of the 
        # target image. This gives us a handy noise map for the resulting
        # difference image.
        hduwt        = fits.PrimaryHDU(diffstd**2)
        # (write the variance to be consistent with what they give us)q
        hduwt.writeto(diffwtoutname, overwrite=True)
        
        if plotintermediate:
            # from matplotlib import rc
            # rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size'   : 10})
            # rc('text', usetex=True)
            # second plotintermediate block, finishing the figure.
            ax3.imshow(newimg,  vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
            ax4.imshow(diffimg, vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
            ax5.imshow(np.pad(k.reshape(shapeini), N+1), origin='lower', cmap=cmap)
            ax6.imshow(x[0][:-1].reshape((N,N)), origin='lower', cmap=cmap)
            # fig.suptitle(f"{N}x{N} kernel reduced $\chi^2$: {redχ2:.2f}")
            ax4.set_title(f'difference image (reduced $\chi^2$: {redχ2:.2f})')
            plt.tight_layout()
            plt.waitforbuttonpress()
        # normalise the difference image so that it is on the same zeropoint as the
        # reference image (to construct lightcurve easily)
        diffimg  /= x[0].sum()
        
        # now write the difference image. It will go along
        # its companion weight map saved earlier.
        hdu        = fits.PrimaryHDU(diffimg)
        hdu.header = targetheader
        hdu.header['redchi2'] = redχ2
        hdu.header['srcfile'] = targetpath
        hdu.writeto(diffoutname, overwrite=True)
        # log what chi2 and seeing we actually had:
        seeing    = targetheader['CHIP.SEEING']
        logChi2AndSeeing(outdir, diffoutname, redχ2, seeing)
        # help the garbage collector since this matrix is huge.
        # I found it sometimes helps to explicitely discard huge objects.
        del(fullmatrix)
    else: 
        # if the difference image already exists and not setting "redo",
        # just load it:
        print(f"      Loading difference {diffoutname}\n")
        diffimg      = fits.open(diffoutname)[0].data
    
    
    diffimgcutout = cropToCenter(diffimg, object_extent//2)
    refcutout     = cropToCenter(refarray, object_extent//2)
    
    if np.sum(np.isnan(diffimgcutout)) > diffimgcutout.size // 10:
        return np.nan, np.nan, redχ2
    # return the intensity of the difference as well as the intensity of the
    # target image for relative comparison
    # (in the future, use sextractor on the cutouts rather than simply summing them)
    return np.nansum(diffimgcutout), np.nansum(refcutout), redχ2
