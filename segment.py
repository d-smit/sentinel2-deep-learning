import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import matplotlib.pyplot as plt

from rasterio import features
from rasterstats import zonal_stats
from shapely.geometry import shape
import json
import geojson
import time

from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb
import tifffile as tiff
import scipy.misc

root_path = os.getcwd()

def create_segment_polygon(tif):
    data = tif.read()
    print("Segmenting S2 image using quickshift")
    segments = felzenszwalb(np.moveaxis(data, 0, -1),  scale=100, sigma=0.5, min_size=50)
    print("Extracting shapes")
    shapes = list(features.shapes(segments.astype(np.int32), transform=tif.transform))
    print("Creating GeoDataFrame from polygon segments.")
    seg_gdf = gpd.GeoDataFrame.from_dict(shapes)
    seg_gdf['geometry'] = seg_gdf[0].apply(lambda x: shape(geojson.loads(json.dumps(x))))
    seg_gdf.crs = tif.crs
    print("Segment CRS: {}".format(seg_gdf.crs))
    return seg_gdf.drop([0, 1], axis=1), segments, shapes

# Background image

def plot_segments(segments):
    print('Plotting segments over scene render...')

    image_to_plot = rio.open(root_path + '/data/masked_image_render.tif')
    data = image_to_plot.read()

    plot_data = data[0:3]  # mark_boundaries needs [width, height, 3] format

    scipy.misc.toimage(plot_data, cmin=0.0).save('masked_plot.tif')
    image_plot = tiff.imread('masked_plot.tif')
    fig, ax = plt.subplots(1, 1, figsize=(25, 25), sharex=True, sharey=True)
    ax.imshow(mark_boundaries(image_plot, segments))
    plt.tight_layout()
    plt.savefig('masked_segmented.tif')
    plt.show()

# Getting segment IDs from segments_df

def get_zones_and_dists(df):
    print('Getting zonal stats for segments over scene...')

    ''' Getting zonal stats for each segment and calculating 
        mean distributions. '''

    st = time.time()
    segment_stats = zonal_stats(df, root_path + '/data/masked.tif',
                                stats = 'count min mean max median')

    zones_df = pd.DataFrame(segment_stats)
    zones_df = zones_df.dropna()
    zones_df['zone_id'] = np.nan

    zone_means = zones_df['mean']

    dists = [10, 25, 50, 75, 90]

    dist_values = []

    for i in range(0, len(dists)):
        d = np.percentile(zone_means, dists[i])
        dist_values.append(d)
    en=time.time()
    print('zone stats in {} sec'.format(en-st))
    return zones_df, tuple(dist_values)

def tag_zones(df, dv):
    print('Tagging zones...')

    ''' Giving each segment a number between 1-5 based on their 
        similarity to the distribution percentiles. '''
    st = time.time()

    for idx, row in df.iterrows():
        if (df.at[idx, 'mean']) < 1.2 * dv[0]:
            df.at[idx, 'zone_id'] = 1

        elif 1.2 * dv[0] < (df.at[idx, 'mean']) < 1.1 * dv[1]:
            df.at[idx, 'zone_id'] = 2

        elif 1.1 * dv[1] < (df.at[idx, 'mean']) < 1.1 * dv[2]:
            df.at[idx, 'zone_id'] = 3

        elif 1.1 * dv[2] < (df.at[idx, 'mean']) < 1.2 * dv[3]:
            df.at[idx, 'zone_id'] = 4

        elif 1.2 * dv[3] < (df.at[idx, 'mean']):
            df.at[idx, 'zone_id'] = 5

    df = df.dropna()
    en=time.time()
    print('zones tagged in {} sec'.format(en-st))
    return df

def match_segment_id(pixel_df, poly_df):
    print('Matching pixels with their segment ID...')

    ''' Parsing through the extracted points and matching
        each pixel with the segment ID of the segment
        that contains it. '''
    st = time.time()

    for i, row in enumerate(pixel_df.itertuples(), 0):
        point = pixel_df.at[i, 'geometry']

        for j in range(len(poly_df)):
              poly = poly_df.iat[j, 0]

              if poly.contains(point):
                  pixel_df.at[i, 'segment_id'] = poly_df.iat[j, 1]
              else:
                  pass
    en=time.time()
    print('pixels and segments matched in {} sec'.format(en-st))
    return pixel_df    # takes ages, trying to make faster with apply


