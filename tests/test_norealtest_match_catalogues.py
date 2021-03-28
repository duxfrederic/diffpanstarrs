#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:08:27 2020

@author: frederic
"""

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord 


def fitsToSkyCoords(fitspath):
    cat = fits.getdata(fitspath)
    cat_array_ra   = np.array([e[0] for e in cat])
    cat_array_dec  = np.array([e[1] for e in cat])
    skycoord       = SkyCoord(ra=cat_array_ra*u.degree, dec=cat_array_dec*u.degree)
    return cat, skycoord

def matchCatalogues(small, bigger, threshold=1e-3):
    idx, d2d, d3d = small.match_to_catalog_sky(bigger)
    matches = np.where(d2d<threshold*u.degree)
    idx, d2d = idx[matches], d2d[matches]
    return idx, d2d


knowns, knowns_coords = fitsToSkyCoords('test_data/news_catalogue/lensedquasars.fits')
contas, contas_coords = fitsToSkyCoords('test_data/news_catalogue/contaminants.fits')
big, big_coords       = fitsToSkyCoords('test_data/news_catalogue/gaia_doubles_w1w2_0.5_RA_0to360_G_16p5_3as_W1_16.5_dec_m30_b15_0p8_to_3as_pmsig_5_wdensities.fits')
medium, medium_coords = fitsToSkyCoords('test_data/news_catalogue/reduced_catalogue.fits')
ext, ext_coords       = fitsToSkyCoords('test_data/news_catalogue/extended_reduced.fits')

#%%
idx, d2d = matchCatalogues(knowns_coords, big_coords)
print(f"{len(idx)} lenses of the {len(knowns)} known ones were found in the gaia cross-match.")
#%%
idx, d2d = matchCatalogues(contas_coords, big_coords)
print(f"{len(idx)} lenses of the {len(contas)} suspected ones were found in the gaia cross-match.")
#%%
idx, d2d = matchCatalogues(knowns_coords, medium_coords)
print(f"{len(idx)} lenses of the {len(knowns)} known ones were found in the reduced gaia cross-match.")
#%%
idx, d2d = matchCatalogues(knowns_coords, ext_coords)
print(f"{len(idx)} lenses of the {len(knowns)} known ones were found in the extended reduced gaia cross-match.")
#%%
idx, d2d = matchCatalogues(contas_coords, medium_coords)
print(f"{len(idx)} lenses of the {len(contas)} suspected ones were found in the reduced gaia cross-match.")
#%%
# now we import a catalogue that we obtained after some manual sorting of the 
# reduced gaia catalogue. 
catalogue = 'test_data/news_catalogue/diffpanstarrs_classified.csv'
ra, dec, lens = np.loadtxt(catalogue, skiprows=1, delimiter=',', usecols=(0,1,2), unpack=1)
ra, dec = ra[np.where(lens)], dec[np.where(lens)]
new_coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)

# sanity check, we find them in the originating catalogue:
idx, d2d = matchCatalogues(new_coords, medium_coords)
print(f"{len(idx)} new lenses of the {len(new_coords)} suspected ones were found in the reduced gaia cross-match (sanity check).")
#%%
# how many are not in the known lenses?
idx, d2d = matchCatalogues(new_coords, knowns_coords)
print(f"{len(idx)} new lenses of the {len(new_coords)} suspected ones were found in the known lenses catalogue.")
# how many are not in the wider, possibly contaminated catalogue?
idx, d2d, d3d = new_coords.match_to_catalog_sky(contas_coords)
matches = np.where(d2d<1e-3*u.degree)
idx, d2d = matchCatalogues(new_coords, contas_coords)
print(f"{len(idx)} new lenses of the {len(new_coords)} suspected ones were found in the known lenses catalogue.")