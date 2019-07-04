import geopandas as gpd
import rasterio as rio
from rasterio import features
from rasterio.merge import merge
from rasterio.plot import show
import pylab as pl
from fiona.crs import from_epsg
from shapely.geometry import box as geobox
from shapely.geometry import shape
import json
import geojson
import numpy as np
import pandas as pd
import os
from glob import glob
from subprocess import check_output
import matplotlib.pyplot as plt
from keras.models import Sequential
import land_classification as lc
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from skimage.segmentation import felzenszwalb, quickshift

# Defining Swindon area of interest

aoi_geo = geobox(-2.29, 51.51, -1.71, 51.61)
aoi = gpd.GeoDataFrame([], geometry=[aoi_geo])
aoi.crs = from_epsg(4326)
aoi.to_file('data/aoi.geojson', driver='GeoJSON')

# Getting land-cover classes 

with open('data/labels.json') as jf:
    names = json.load(jf)
    
root_path = check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode()

# Reading and merging DEM data

def merge_dem():

    files = glob(root_path + '/data/Ancillary/swindon*', recursive=True)
    files.sort()
    tifs = list(map(rio.open, files))
    dem_data = pl.stack(list(map(lambda x: x.read(1).astype(pl.int16), tifs)))
    dem_profile = tifs[0].profile
    return dem_data, dem_profile

dem_data, dem_profile = merge_dem()

# Reading and merging band data

s2_band = 'S2A.SAFE'
data, profile = lc.merge_bands(s2_band, res='10')

# Writing and masking band raster

lc.write_raster('data/swindon/merged.tif', data, profile)
lc.mask_raster(aoi, 'data/swindon/merged.tif', 'data/swindon/masked.tif')

## Writing and masking DEM raster 

lc.write_raster('data/swindon/merged_dem.tif', dem_data, dem_profile)
lc.mask_raster(aoi, 'data/swindon/merged_dem.tif', 'data/swindon/masked_dem.tif')

# Making mosaic of both

masks = os.path.join(root_path, 'data/swindon/masked*.tif')

bands_dems = glob(masks)

band_ancillary_mosaic = []

for tif in bands_dems:
    src = rio.open(tif)
    band_ancillary_mosaic.append(src)

mosaic, out_trans = merge(band_ancillary_mosaic, indexes=range(12))