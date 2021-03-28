#!/usr/b3in/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:24:59 2020

@author: frederic


At first, this was supposed to sort effectively for lenses.
However it is not as simple as I anticipated, thus this will attribute a score
to the variability maps that exhibit something at all.

The higher the score, the more likely it is to be shown to a human classifier
(me) at https://diffpanstarrs.space

"""
from      os.path                         import  dirname, join
import    numpy                           as      np 
from      astropy.io                      import  fits
import    warnings 
from      astropy.utils.exceptions        import  AstropyWarning
warnings.simplefilter('ignore', AstropyWarning)
# import a sigmoid function since we'll compute some pseudo scores:
from      keras.models                    import  load_model
from      keras                           import  backend
installation_dir = dirname(__file__)
model = load_model(join(installation_dir, 'cnn_weights', 'model.hdf5'))


from diffpanstarrs.imageMoments  import imageFindPatches, imageEllipticity, getGrid

def normalize(images):
    images[np.isnan(images)] = 0
    if len(images.shape) == 2:
        images = np.array([images])
    for i in range(len(images)):
        m, s = np.mean(images[i]), np.std(images[i])
        N = 10
        images[i][images[i]>m+N*s] = m+N*s
        images[i] -= np.min(images[i])
        images[i] /= np.max(images[i])
    return images

def decide(image):
    if type(image) is str:
        data = fits.getdata(image)
    else:
        data = image
    data = normalize(data)
    l, sx, sy = data.shape 
    data = data[:, sx//2-20:sx//2+20, sy//2-20:sy//2+20]
    if backend.image_data_format() == 'channels_first':
        data = data.reshape(data.shape[0], 1, 40, 40)
    else:
        data = data.reshape(data.shape[0], 40, 40, 1)
    return model.predict(data)[0][0]
    
    
if __name__ == "__main__":
    decide('hello')

def decide_naive(normimage, bigoriginalstack=None, debug=False, logger=None, redo_sextractor=True):
    score  = 0
    score_frac = score_multi = score_one = score_none = score_toobright = 0
    image  = fits.open(normimage)[0].data
    # dimage = fits.open(str(normimage).replace('.fits', '.wt.fits'))[0].data**0.5
    
    # first, check if the variability image lacks too much data:
    # X,Y are cartesian coordinates with origin at the center of the image
    X, Y = getGrid(image.shape)
    radialgrid = (X**2+Y**2)**0.5
    # we care the most if there are nans in the center!
    radialgrid = radialgrid.max() - radialgrid
    frac = np.sum(np.isnan(image).astype(np.float)*radialgrid) / radialgrid.sum()
    if frac > 0.2:
        score_frac = -frac
        score += score_frac
    isoflux, (x,y), (a,b), segmap = imageFindPatches(normimage, redo=redo_sextractor, insane_blur=False)
    limit_ratio = 2
    if len(isoflux) == 0:
        score_none = -1
    else:
        # sort through the detections: if a detection as a ratio a/b of more than 2, bad. 
        # (probably a trail or a border of a nan region)
        
        where = np.where( (a/b < limit_ratio) * (a/b) > 1/limit_ratio)
        
        isoflux, x,y = isoflux[where], x[where], y[where]
        if len(isoflux) > 1:
            sortedfluxes = sorted(isoflux)
            
            if sortedfluxes[1] > 0.2 * sortedfluxes[0]:
                score_multi = 1
                score += score_multi
            else: 
                score_one = 0.5
        elif len(isoflux) == 1:
            score_one = 0.3
            score += score_one
            # if max(a[0] / b[0], b[0]/a[0]) > 1.5:
                # score += 0.3
            # else:
            image = fits.getdata(normimage)
            seg   = fits.getdata(normimage.replace('.fits', '_seg.fits'))
            mask = seg != 0 
            el = imageEllipticity(image*mask)[0]
            if el > 0.02:
                score += 0.3 + max(0, 10*(el-0.02))
                print(el)
            if max(isoflux) < 0.005:
                score -= 1
            
        elif len(isoflux) == 0:
            score_none = -1

    score += score_none
    
    if bigoriginalstack:
        # Now, if the central object is the brightest in the field, then a penalty must be added:
        # the psf structure will not have been determined ideally form the lesser other sources. 
        # this might results in artifical residuals that will have been detectd by sextractor.
        isoflux, (x,y), (a,b), segmap = imageFindPatches(bigoriginalstack, redo=redo_sextractor, insane_blur=False)
        shape = fits.getdata(bigoriginalstack).shape
        # square images!!
        where = np.where( (a/b < limit_ratio) * (a/b) > 1/limit_ratio)
        isoflux, x, y = isoflux[where], x[where], y[where]
        center = shape[0]//2
        objects_within_center = np.where( (np.abs(x-center)<20) * (np.abs(y-center)<20) )
        wherenot = np.where( ~ ( (np.abs(x-center)<20) * (np.abs(y-center)<20) ) )
        isoflux_notcenter = isoflux[wherenot]
        if len(objects_within_center[0]) == 0 or len(wherenot[0]) == 0:
            pass
        else:
            if max(isoflux[objects_within_center]) > 1.2*max(isoflux_notcenter):
                score_toobright = - 0.1*(max(isoflux_notcenter) / max(isoflux[objects_within_center]) -1)
                score += score_toobright
        
    if debug:
        print(f"nan penalty: {score_frac:.1f}   multi boost: {score_multi}   one boost: {score_one}   penalty none: {score_none}   penalty too bright: {score_toobright}")
    if score > 5:
        score = -1
    return score 
    """
    mask = segmap != 0
    el = imageEllipticity(image*mask)[0]
    # 50 and -5 are guesses, we'll need to fit them to maximize our decision power.
    el_score = expit(el*50-5)
    score += el_score
    
    # moffat ratio:
    params, _ = fitMoffatProfile(image, dimage, mask)
    a, b      = params[5:7]
    ratio     = max(a,b)/min(a,b)
    ratio_score = expit(ratio*10-15)
    # the ratio can often be off, weigh this less:
    score += 0.6*ratio_score
    
    # residuals:
    model = moffatProfile(*params, image.shape)
    res   = image-model
    res   = res[mask]
    res   = res[~np.isnan(res)]
    if len(res) >= 8:
        k, p  = normaltest(res)
        res_score = expit(-np.log10(p) - 2.5)
        score += res_score
    else:
        res_score = -0.1
        score += res_score
        p = np.nan
    if debug:
        m1 = f"  values:   ell: {el:.02f}, ratio: {ratio:.02f}, res: {-np.log10(p):.02f}"
        m2 = f"  scores:   ell: {el_score:.02f}, ratio: {ratio_score:.02f}, res: {res_score:.02f}"
        print(m1) 
        print(m2)
        if logger:
            logger.info(m1)
            logger.info(m2)
    return score, (el_score, ratio_score, res_score)
    """

