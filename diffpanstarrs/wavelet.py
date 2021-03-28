#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 11:34:22 2019

@author: frederic dux
"""

import numpy               as     np
from   scipy.signal        import convolve2d
from   skimage.restoration import denoise_tv_chambolle
###############################################################################
################## Some scaling functions #####################################
###############################################################################
# B2 spline : (gaussian approximation)
f1D_B2 = np.array([ 1./16, 1./4, 3./8, 1./4, 1./16 ])

# Daubechies-4
c0 = (1+3**0.5) / (4*2**0.5)
c1 = (3+3**0.5) / (4*2**0.5)
c2 = (3-3**0.5) / (4*2**0.5)
c3 = (1-3**0.5) / (4*2**0.5)
f1D_D4 = [c0, c1, c2, c3]; f1D_D4 = f1D_D4/np.sum(f1D_D4)

# Daubechies-6
f1D_D6 = [0.47, 1.141, 0.65, -0.191, -0.121, 0.0498]
f1D_D6 = np.array(f1D_D6)/2
# Daubechies-2 = Haar
f1D_Haar = [0.5,0.5]
# linear spline:
cs =    [1./4, 1./2, 1./4]
f1D_lin = np.array(cs)

# gaussian:
f1D_gauss = [0.333, 0.3333, 0.3333]

scaling_functions = {
        'B2-spline' : f1D_B2,
        'Daubechies-4': f1D_D4, 
        'Daubechies-6': f1D_D6, 
        'Haar': f1D_Haar, 
        'linear-spline': f1D_lin,
        'gaussian': f1D_gauss
}
modes = ['linear', 'dyadic']
###############################################################################
###############################################################################
###############################################################################


def stack_2d_bspline_smooth(Imag, Num_Plan, scaling_function, mode):
    """
          a trou smoothing with the scaling function. 
    """
    lenscaling = len(scaling_function)
    if mode == 'linear':
        step = max(1,Num_Plan)
    elif mode == 'dyadic':
        step = 2**Num_Plan
        size = step * lenscaling
    else :
        raise(ValueError)
    size                  = step * lenscaling
    
    f1D_atrou             = np.zeros(size)
    f1D_atrou[:size:step] = scaling_function[:]
    starlet               = (np.kron(f1D_atrou,f1D_atrou).reshape((size,size))) 
    smooth                = convolve2d(Imag, starlet, mode='same')
    return smooth


def imageStarletTransform(img, Nbr_Plan, scaling_function='Haar', mode='dyadic'):
    """
    Starlet transform of the image contained in the 2D array img.
    Decomposes the information onto Nbr_Plan images of the same size.

    Parameters
    ----------
    img : 2D numpy array
        the image to be decomposed.
    Nbr_Plan : int
        number of levels of details.
    scaling_function : str, optional
        choice of the scaling function, can be:
            -  'B2-spline' 
            -  'Daubechies-4'
            -  'Daubechies-6'
            -  'Haar'  (corresponds to Daubechies-2)
            -  'linear-spline'
            -  'gaussian': f1D_gauss
        The default is 'Haar'.
    mode : str, optional
        linear or dyadic. The latter corresponds to the more rigorous
        definition of an a trou wavelet transform. But the former works
        just as well as is faster. The default is 'dyadic'.
        (dyadic: the scaling function grows in size exponentially with the number of levels.)
         linear: the scaling function grows linearly with the number of levels.)

    Returns
    -------
    stack : list
        list of 'Nbr_Plan' 2D numpy arrays containing the different layers.
        The original image is simply obtained with sum(stack).
        stack[0] contains the finer details, while stack[-1] contains the
        residuals.
    """
    scaling_function = scaling_functions[scaling_function]
    img   = np.copy(img)
    stack = []
    for Num_Plan in range(Nbr_Plan):
        Plan   = np.zeros_like(img)
        smooth = stack_2d_bspline_smooth(img, Num_Plan, scaling_function, mode)
        Plan   = img - smooth
        img    = smooth
        stack.append(Plan)
    stack.append(img)
    return stack

    
def waveletDenoise(image, weight=2):
    noise, residuals = imageStarletTransform(image, 1, scaling_function='B2-spline')
    noise[np.isnan(noise)] = np.nanpercentile(noise, 10)
    denoised         = denoise_tv_chambolle(noise, weight=2)
    return residuals + denoised