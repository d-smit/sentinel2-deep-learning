#!/usr/bin/env python3

import geopandas as gpd
from fiona.crs import from_epsg
from shapely.geometry import box as geobox
import json

import land_classification.land_classification as lc

<<<<<<< HEAD
aoi_geo = geobox(-2.29, 51.51, -1.71, 51.61)
aoi = gpd.GeoDataFrame([], geometry=[aoi_geo])
aoi.crs = from_epsg(4326)
aoi.to_file('/home/david/Uni/Thesis/lc_gsi/data/aoi.geojson', driver='GeoJSON')
#
with open('/home/david/Uni/Thesis/lc_gsi/data/labels.json') as jf:
=======
aoi = False

if not aoi:
    aoi_geo = geobox(-2.29, 51.51, -1.71, 51.61)
    aoi = gpd.GeoDataFrame([], geometry=[aoi_geo])
    aoi.crs = from_epsg(4326)
    aoi.to_file('./data/aoi.geojson', driver='GeoJSON')
else:
    aoi = gpd.read_file('/data/aoi.geojson')

with open('./data/labels.json') as jf:
>>>>>>> 56b27e3522e7cb8f429974aa3fd4afd0f2628ee1
    names = json.load(jf)

s2_band = 'S2A.SAFE'
data, profile = lc.merge_bands(s2_band, res='10')
lc.write_raster('data/merged.tif', data, profile)

lc.mask_raster(aoi, 'data/merged.tif', 'data/masked.tif')

bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08']
pe = lc.PointExtractor(aoi)
points_df = pe.get_n(300)
print(points_df)
points_df = lc.sample_raster(points_df, 'data/Corine_Orig_10m_OS_AoI1.tif', bands=['labels'])
points_df = lc.sample_raster(points_df, 'data/masked.tif', bands=bands)
print(points_df.columns)
clean_df = lc.remove_outliers(points_df, bands=bands, indices=False)
clean_df = lc.create_zero_samples(clean_df, bands=bands)
clean_df = lc.calc_indices(clean_df)

pred, proba, cm, cls = lc.classify(clean_df, onehot=False, labels=names)
# print(pred)
# print(proba)
