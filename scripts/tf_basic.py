#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:45:24 2019

@author: david
"""
import geopandas as gpd
import rasterio as rio
import pylab as pl
import json
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras import backend as K
from fiona.crs import from_epsg
from shapely.geometry import box as geobox
import land_classification as lc
from sklearn.model_selection import train_test_split

aoi_geo = geobox(-2.29, 51.51, -1.71, 51.61)
aoi = gpd.GeoDataFrame([], geometry=[aoi_geo])
aoi.crs = from_epsg(4326)
aoi.to_file('data/aoi.geojson', driver='GeoJSON')

with open('data/labels.json') as jf:
    names = json.load(jf)
    
s2_band = 'S2A.SAFE'

data, profile = lc.merge_bands(s2_band, res='10')

data = data[1]

lc.write_raster('data/merged.tif', data, profile)
lc.mask_raster(aoi, 'data/merged.tif', 'data/masked.tif')


pe = lc.PointExtractor(aoi)
 
points_df = pe.get_n(500)
bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08']
 
points_df = lc.sample_raster(points_df, 'data/Corine_S2_Proj_2.tif', bands=['labels'])
points_df = lc.sample_raster(points_df, 'data/masked.tif', bands=bands)
 
clean_df = lc.remove_outliers(points_df, bands=bands, indices=False)
clean_df = lc.calc_indices(clean_df)
 
class_cols = 'labels_1'
 
temp_bands = ['B01_1', 'B02_1', 'B03_1', 'B04_1', 'B05_1', 'B06_1', 'B07_1', 'B08_1']
 
X = clean_df[temp_bands]
X = X.values
y = clean_df[class_cols]
y = y.values
 
mask_src = rio.open('data/masked.tif')
 
profile = mask_src.profile
data = mask_src.read(list(pl.arange(mask_src.count) + 1))
gdf = lc.create_raster_df(data, bands=temp_bands)
gdf = lc.calc_indices(gdf)
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
model = Sequential()
 
model.add(Conv2D(64, 2, padding='same', activation='relu'))
 
model.fit(X_train, y_train,
       batch_size=1,
       epochs=1,
       verbose=1,
       validation_split=0.1)

