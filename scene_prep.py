import os
from shapely.geometry import box as geobox
from fiona.crs import from_epsg
import geopandas as gpd
import rasterio as rio
import json
import land_classification as lc
# import segment as seg
import numpy as np

root_path = os.getcwd()

Segment = False

def prepare_scene():
    aoi_geo = geobox(-2.29, 51.51, -1.71, 51.61)
    aoi = gpd.GeoDataFrame([], geometry=[aoi_geo])
    aoi.crs = from_epsg(4326)
    aoi.to_file('data/aoi3.geojson', driver='GeoJSON')
    
    with open('data/labels.json') as jf:
        names = json.load(jf)
    s2_band = 'S2A.SAFE'

    data, profile = lc.merge_bands(s2_band, res='10')
    
    lc.write_raster('data/merged.tif', data, profile)
    lc.mask_raster(aoi, 'data/merged.tif', 'data/masked.tif')

    return data, profile, names, aoi

# def segment_scene(tif):
#     seg_df, segments, shapes = seg.create_segment_polygon(tif)
#     seg.plot_segments(segments)
#     zones_df, dists = seg.get_zones_and_dists(seg_df)
#     zones_df = seg.tag_zones(zones_df, dists)
#     seg_df['zone_id'] = zones_df['zone_id']

#     return seg_df

tif = rio.open('data/masked.tif', 'r')
# seg_df = segment_scene(tif)

def create_df(df, bands= ['B02', 'B03', 'B04', 'B08']):

    points_df, values = lc.sample_raster(df, 'data/Corine_S2_Proj_2.tif', bands=['labels'])

    # if Segment:
    #     points_df = seg.match_segment_id(points_df, seg_df)
    #     points_df = lc.sample_raster(points_df, 'data/masked.tif', bands=bands)
    # else:
    points_df, values = lc.sample_raster(points_df, 'data/masked.tif', bands=bands)

    points_df.to_file('points_shp_exj.json', driver='GeoJSON')

    points_df.drop(['geometry'], axis=1)
    points_df.to_csv('points.csv')
    # values = values.astype(float)
    # print(values)
    np.savez_compressed('patch_arrays.npz', values)

    print('pixel df stored')

    return points_df, values

data, profile, names, aoi = prepare_scene()
pe = lc.PointExtractor(aoi)
points_df = pe.get_n(100000)
print('Grabbing pixel patches...')
df, values = create_df(points_df)
