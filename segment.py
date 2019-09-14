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

from tqdm import tqdm

root_path = os.getcwd()

'''
We use skimage to apply graph-based segmentation to our scene tif. We turn segments 
into polygons, and add them to a GeoDataframe. This retains their spatial context as 
we also have their coordinates. This will come in use later. 
'''

def create_segment_polygon(tif):
    data = tif.read()
    print("Segmenting...")
    segments = felzenszwalb(np.moveaxis(data, 0, -1),  scale=1200, sigma=0.5, min_size=500)
    print("Extracting shapes")
    shapes = list(features.shapes(segments.astype(np.int32), transform=tif.transform))
    print("Creating GeoDataFrame from polygon segments.")
    seg_gdf = gpd.GeoDataFrame.from_dict(shapes)
    seg_gdf['geometry'] = seg_gdf[0].apply(lambda x: shape(geojson.loads(json.dumps(x))))
    seg_gdf.crs = tif.crs
    print("Segment CRS: {}".format(seg_gdf.crs))
    return seg_gdf.drop([0, 1], axis=1), segments, shapes

def plot_segments(segments):
    print('Plotting segments over scene render...')

    image_to_plot = rio.open(root_path + '/data/masked_render.tif')
    data = image_to_plot.read()
    plot_data = data[0:3]  # mark_boundaries needs [width, height, 3] format
    scipy.misc.toimage(plot_data, cmin=0.0).save('masked_plot.tif')
    image_plot = tiff.imread('masked_plot.tif')
    fig, ax = plt.subplots(1, 1, figsize=(25, 25), sharex=True, sharey=True)
    ax.axis('off')
    ax.imshow(mark_boundaries(image_plot, segments))
    plt.tight_layout()
    plt.savefig('segment_300_05_s250.png')
    plt.show()

'''
Next, we want to find out the characteristics of the segments. To do this, 
we run zonal_stats, which gives us statistics for each segment. Next, we create
a dataframe using the statistics of each segment as a row. We choose mean as our 
main characteristic, and calculate the 10th, 25th, 50th, 75th and 90th percentile
values.
'''

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
    print('zone stats in {} sec'.format(int(en-st)))
    return zones_df, tuple(dist_values)

'''
Now we want to see how an individual segment compares to the global averages.
To do this, we compare each segment mean to the percentile means. Using a boundary 
around each value, we attribute "zone_id" to each segment. If a given segment
is within the 10th percentile boundary, a 1 is added for its zone_id, for example.
'''

def tag_zones(df, dv):
    print('Tagging zones...')

    ''' Giving each segment a number between 1-5 based on their 
        similarity to the distribution percentiles. '''

    st = time.time()

    for idx, row in df.iterrows():
        if (df.at[idx, 'mean']) <= 1 * dv[0]:
            df.at[idx, 'zone_id'] = 1

        elif 1 * dv[0] < (df.at[idx, 'mean']) <= 1 * dv[1]:
            df.at[idx, 'zone_id'] = 2

        elif 1 * dv[1] < (df.at[idx, 'mean']) <= 1 * dv[2]:
            df.at[idx, 'zone_id'] = 3

        elif 1 * dv[2] < (df.at[idx, 'mean']) <= 1 * dv[3]:
            df.at[idx, 'zone_id'] = 4

        elif 1 * dv[3] < (df.at[idx, 'mean']) <= 1 * dv[4]:
            df.at[idx, 'zone_id'] = 5

        elif 1 * dv[4] < (df.at[idx, 'mean']):
            df.at[idx, 'zone_id'] = 6

    df = df.dropna()
    en=time.time()
    print('zones tagged in {} sec'.format(float(en-st)))
    return df

'''
Finally, we want to apply this knowledge of local regions on a pixel-level.
This function parses through our pixels which were previously extracted, finds 
the segment it's in and attaches the respective segment_id value to the pixel. 
This is done for every pixel in our dataset. 
'''


def match_segment_id(pixel_df, poly_df, poly_df_zones):
    print('Matching pixels with their segment ID...')

    res = np.empty(pixel_df.geometry.shape)

    for i, point in tqdm(enumerate(pixel_df.geometry.values)):
        for j, poly in enumerate(poly_df.geometry.values):
              if poly.contains(point):
                  res[i] = poly_df_zones[j]
              else:
                  pass

    pixel_df['segment_id'] = res.tolist()
    pixel_df = pixel_df.dropna()
    return pixel_df


