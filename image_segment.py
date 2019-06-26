#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 19:25:21 2019

@author: david
"""

import rasterio as rio
import matplotlib.pyplot as plt
import skimage 
import pylab as pl
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import land_classification as lc

tif = rio.open('data/swindon/masked.tif')

data = pl.stack(list((lambda x: x.read(1).astype(pl.int16), tif)))

s2_band = 'S2A.SAFE'

data, profile = lc.merge_bands(s2_band, res='10')

tif = data[0:3]

tif = tif.reshape(tif.shape[1], tif.shape[2], tif.shape[0])

segments_quick = quickshift(tif, kernel_size=3, max_dist=6, ratio=0.5)

