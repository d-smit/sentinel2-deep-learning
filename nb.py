import geopandas as gpd
import pylab as pl
import rasterio as rio
import json
from geopandas import GeoDataFrame
from fiona.crs import from_epsg
from shapely.geometry import box as geobox

import land_classification.land_classification as lc

# Designate an area of interest to capture from S2 data

aoi_geo = geobox(-2.29, 51.51, -1.71, 51.61)
aoi = gpd.GeoDataFrame([], geometry=[aoi_geo])
aoi.crs = from_epsg(4326)
aoi.to_file('data/aoi.geojson', driver='GeoJSON')

# Accessing labels to be used later

with open('data/labels.json') as jf:
    names = json.load(jf)

# Sentinel-2 data directory

s2_band = 'S2A.SAFE'

# Combining each band into one multi-dimensional numpy array of uniform resolution

data, profile = lc.merge_bands(s2_band, res='10')

# Gives shape (8, 10980, 10980) where each "level" is an array representing the pixel values of that band

# Writing this as a multiband raster

lc.write_raster('data/merged.tif', data, profile)

# Masking or portioning this raster using the aoi created earlier.

lc.mask_raster(aoi, 'data/merged.tif', 'data/masked.tif')

# Opening both merged and masked in QGIS shows the area of interest.

# Using the projection in the 10m Corine on the S2 tif file.

'''Using the PointExtractor to get a 
    selection of points - this 
    seems to get around the problem of having 
    different resolution between s2 data and 
    Corine data 
'''

pe = lc.PointExtractor(aoi)
points_df = pe.get_n(25)
print(points_df)

# Point values are all 0

bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08']

''' Original choice of 4 bands gave shape 
    errors when creating sample raster.
    Assumed this was because original 
    masked raster was made up 8 levels as
    raw data had 8 bands. Therefore decided 
    to keep all 8 bands. '''


# points_df = lc.sample_raster(points_df, 'data/Corine_Orig_10m_OS_AoI1.tif', bands=['labels'])
# print(points_df)

points_df = lc.sample_raster(points_df, 'data/Corine_S2_Proj_2.tif', bands=['labels'])
print(points_df)

''' Sample raster function reads the extracted points and gets the corresponding 
Corine pixel value, labelling the dataframe. '''

points_df = lc.sample_raster(points_df, 'data/masked.tif', bands=bands)

''' Used again here to get each extracted points corresponding band value. 
'''
print(points_df.columns)

clean_df = lc.remove_outliers(points_df, bands=bands, indices=False)

print(clean_df.isnull().values.any())
print(clean_df)

# Below line doesn't work

clean_df = lc.create_zero_samples(clean_df)

clean_df = lc.calc_indices(clean_df)

print(clean_df)

pred, proba, cm, cls = lc.classify(clean_df, onehot=False, labels=names)

print('Predictions: \n{}'.format(pred))
print('Probabilities: \n{}'.format(proba))
print('Confusion Matrix: \n{}'.format(cm))
print('Classifier parameters: \n{}'.format(cls))


# Moving onto stuff in notebook not in this example

# Below line only neccesary if PointExtractor is not used?

# lc.copy_projection('data/masked.tif', 'data/Corine_Orig_10m_OS_AoI1.tif', 'data/Corine_S2_Proj_2.tif')

mask_src = rio.open('data/masked.tif')

# Within .ml classify function

profile = mask_src.profile

# This data object is confusing - seems to be arrays of just -9999s

data = mask_src.read(list(pl.arange(mask_src.count) + 1));print(data)
c_tif = rio.open('data/Corine_S2_Proj_2.tif')
c_data = c_tif.read()
print(list(pl.arange(mask_src.count) + 1))
print('Rows in data: {}'.format(len(data)))
print('Columns in data: {}'.format(len(data[0])))
print('Data shape: {}'.format(data.shape))
print(profile)

gdf = lc.create_raster_df(data)
gdf = lc.calc_indices(gdf)
gdf['labels'] = lc.create_raster_df(pred_array=c_data, bands=['labels'])['labels']
print(gdf)

# win_res = 10 ## Window resolution of 10 implies 100m pixels to predict
# data_30 = pl.zeros(shape=(data.shape[0], int(data.shape[1]/win_res), int(data.shape[1]/win_res), win_res, win_res))
# print(data_30)
