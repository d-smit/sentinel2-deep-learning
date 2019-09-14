import os
from shapely.geometry import box as geobox
from fiona.crs import from_epsg
import geopandas as gpd
import rasterio as rio
import json
from land_classification.raster import merge_bands, write_raster, mask_raster
from land_classification.sampling import sample_raster, PointExtractor
import segment as seg
import numpy as np

root_path = os.getcwd()

Segment = False

'''
First, we want to load in our scene. Using Sentinel-2 data collected
by GSI, we access the relevant resolution files, reading each band individually
and creating our 3-dimensional raster. This raster represents a large scene, which we want
to crop using our area of interest geometries (aoi). 
'''

def prepare_scene():
    aoi_geo = geobox(-2.29, 51.51, -1.71, 51.61)
    aoi = gpd.GeoDataFrame([], geometry=[aoi_geo])
    aoi.crs = from_epsg(4326)
    aoi.to_file('data/aoi.geojson', driver='GeoJSON')
    
    with open('data/labels.json') as jf:
        names = json.load(jf)
    s2_band = 'S2A.SAFE'

    data, profile = merge_bands(s2_band, res='10')
    
    write_raster('data/merged.tif', data, profile)
    mask_raster(aoi, 'data/merged.tif', 'data/masked.tif')

    return data, profile, names, aoi

tif = rio.open('data/masked2.tif', 'r')

'''
Next, if we are carrying out our segmentation approach, we'll call the below functions,
taken from segment.py. These functionalities are described in full in segment.py.,
we segment our scene, plot it for visualization, run zonal statistics, tag each zone
based on mean value and then create a dataframe from these values.  
'''

def segment_scene(tif):
    seg_df, segments, shapes = seg.create_segment_polygon(tif)
    seg.plot_segments(segments)
    zones_df, dists = seg.get_zones_and_dists(seg_df)
    zones_df = seg.tag_zones(zones_df, dists)
    seg_df['zone_id'] = zones_df['zone_id']
    return seg_df

if Segment:
    seg_df = segment_scene(tif)

'''
Here, we extract the band values and labels for our pixels. If we are segmenting, 
sample_raster is set to a buffer size of 0. If we are using the patch approach,
buffer size is set to the desired patch size. 
'''

def create_df(df, bands= ['B02', 'B03', 'B04', 'B08']):

    points_df, values, buffer = sample_raster(df, 'data/Corine_S2_Proj_2.tif', bands=['labels'])

    if Segment:
        print('segmenting')

        '''This adds the segment id to every pixel we've extracted,
        based on which segment contains it. Then adds pixel band 
        values afterwards.'''

        points_df = seg.match_segment_id(points_df, seg_df, seg_df.zone_id.values)
        points_df, values, buffer = sample_raster(points_df, 'data/masked2.tif', bands=bands)
    else:
        points_df, values, buffer = sample_raster(points_df, 'data/masked2.tif', bands=bands)

    values = values[:len(points_df)]

    patch_size = ((2*buffer) + 1)

    print('Patch dimensions: {} by {}'.format(patch_size, patch_size))

    if Segment:
        points_df.to_csv('seg_points_100k.csv')
    else:
        points_df.to_csv('points_400k_{}x{}.csv'.format(patch_size, patch_size))

    np.savez_compressed('patch_arrays_400k_{}x{}.npz'.format(patch_size, patch_size), values)

    print('pixel df stored')

    return points_df, values

data, profile, names, aoi = prepare_scene()

pe = PointExtractor(aoi)
points_df = pe.get_n(400000)
print('Grabbing pixel patches...')
df, values = create_df(points_df)
