#!/usr/bin/env python3

import geopandas as gpd
from fiona.crs import from_epsg
from shapely.geometry import box as geobox
import json

import land_classification as lc

aoi = False

if not aoi:
    aoi_geo = geobox(-2.29, 51.51, -1.71, 51.61)
    aoi = gpd.GeoDataFrame([], geometry=[aoi_geo])
    aoi.crs = from_epsg(4326)
    aoi.to_file('./data/aoi.geojson', driver='GeoJSON')
else:
    aoi = gpd.read_file('/data/aoi.geojson')

with open('./data/labels.json') as jf:
    names = json.load(jf)

s2_band = 'S2A_MSIL2A_20171128T111411_N0206_R137_T30UWC_20171128T130743.SAFE'
data, profile = lc.merge_bands(s2_band)
lc.write_raster('data/merged.tif', data, profile)

lc.mask_raster(aoi, 'data/merged.tif', 'data/masked.tif')

bands = ['B02', 'B03', 'B04', 'B08']
pe = lc.PointExtractor(aoi)
points_df = pe.get_n(30000)

points_df = lc.sample_raster(points_df, 'data/Corine_Orig_10m_OS_AoI1.tif', bands=['labels'])
points_df = lc.sample_raster(points_df, 'data/masked.tif', bands = bands)

clean_df = lc.remove_outliers(points_df, indices=False)
clean_df = lc.create_zero_samples(clean_df)
clean_df = lc.calc_indices(clean_df)

lc.classify(clean_df, onehot=False, labels=names)
