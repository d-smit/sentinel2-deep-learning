import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio import features
from fiona.crs import from_epsg
from shapely.geometry import box as geobox
from shapely.geometry import shape
import json
import geojson
from subprocess import check_output
import matplotlib.pyplot as plt
import land_classification as lc
from skimage.io import imread
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb
import tifffile as tiff
import scipy.misc
from rasterstats import zonal_stats
from itertools import chain

# Defining Swindon area of interest

aoi_geo = geobox(-2.29, 51.51, -1.71, 51.61)
aoi = gpd.GeoDataFrame([], geometry=[aoi_geo])
aoi.crs = from_epsg(4326)
aoi.to_file('data/aoi.geojson', driver='GeoJSON')

# Getting land-cover classes 

with open('data/labels.json') as jf:
    names = json.load(jf)
    
root_path = check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode()

# Reading and merging band data

s2_band = 'S2A.SAFE'
data, profile = lc.merge_bands(s2_band, res='10')

# Writing and masking band raster

lc.write_raster('data/swindon/merged.tif', data, profile)
lc.mask_raster(aoi, 'data/swindon/merged.tif', 'data/swindon/masked.tif')

def create_segment_polygon(tif):
    data = tif.read()
    # data = data[0:3]
    print("Segmenting S2 image using quickshift")

    # segments = quickshift(np.moveaxis(data, 0, -1), kernel_size=3, convert2lab=False, max_dist=6, ratio=0.8)

    '''Creating array where each entry is an array the 
    width of the pic showing where boundaries are '''

    segments = felzenszwalb(np.moveaxis(data, 0, -1),  scale=100, sigma=0.5, min_size=50)
    print("Extracting shapes")

    '''Using rasterio shape function to generate shapes
    based on these pixel values representing boundary lines '''

    shapes = list(features.shapes(segments.astype(np.int32), transform=tif.transform))

    print("Creating GeoDataFrame from polygon segments.")

    ''' Making a GeoDataFrame of geometries from this
        dictionary of shapes and co-ordinates. '''

    seg_gdf = gpd.GeoDataFrame.from_dict(shapes)

    ''' In order to get WKT geometries, need to first
        convert to json, then to geojson, then to WKT. '''

    seg_gdf['geometry'] = seg_gdf[0].apply(lambda x: shape(geojson.loads(json.dumps(x))))
    seg_gdf.crs = tif.crs
    print("Segment CRS: {}".format(seg_gdf.crs))
    return seg_gdf.drop([0, 1], axis=1), segments, shapes

tif = rio.open('data/swindon/masked.tif')

segment_df, segments, shapes = create_segment_polygon(tif)

# Background image

image_to_plot = rio.open('data/segment/masked_image_render.tif')
data = image_to_plot.read()

# mark_boundaries needs three channel image to impose segments over

plot_data = data[0:3]
scipy.misc.toimage(plot_data, cmin=0.0).save('masked_plot.tif')
image_plot = tiff.imread('masked_plot.tif')

fig, ax = plt.subplots(1, 1, figsize=(25, 25), sharex=True, sharey=True)
ax.imshow(mark_boundaries(image_plot, segments))
plt.tight_layout()
plt.show()

# Getting segment IDs from segments_df

segment_stats = zonal_stats(segment_df, 'data/swindon/masked.tif',
            stats = 'count min mean max median')

# list where each element in as a dictionary
    # each dictionary key is a statistic

seg_stats_df = pd.DataFrame(segment_stats)
seg_stats_df = seg_stats_df.dropna()

for seg in segment_stats:
    for key in sorted(seg.keys()):
        if seg['median'] > 1:

#next((item for item in segment_stats if item['median'] == ), None)
        #print('{}: {}'.format(key, seg[key]))

