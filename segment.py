import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features
from fiona.crs import from_epsg
from shapely.geometry import box as geobox
from shapely.geometry import shape
import json
import geojson
import numpy as np
from subprocess import check_output
import matplotlib.pyplot as plt
import land_classification as lc
from skimage.io import imread
from skimage.segmentation import mark_boundaries
from skimage.segmentation import quickshift, felzenszwalb
import tifffile as tiff


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
    #data = data[0:3]
    print("Segmenting S2 image using quickshift")
    # segments = quickshift(np.moveaxis(data, 0, -1), kernel_size=3, convert2lab=False, max_dist=6, ratio=0.8)
    segments = felzenszwalb(np.moveaxis(data, 0, -1),  scale=100, sigma=0.5, min_size=50)
    print("Extracting shapes")
    shapes = list(features.shapes(segments.astype(np.int32), transform=tif.transform))
    print("Creating GeoDataFrame from polygon segments.")
    seg_gdf = gpd.GeoDataFrame.from_dict(shapes)
    seg_gdf['geometry'] = seg_gdf[0].apply(lambda x: shape(geojson.loads(json.dumps(x))))
    seg_gdf.crs = tif.crs
    print("Segment CRS: {}".format(seg_gdf.crs))
    return seg_gdf.drop([0, 1], axis=1), segments

tif = rio.open('data/swindon/masked.tif')

segment_df, segments = create_segment_polygon(tif)

fig, ax = plt.subplots(1, 1, figsize=(25, 25), sharex=True, sharey=True)

data = tif.read()
image_array = data[0:3]
image_array = image_array.reshape(image_array.shape[1],
                                  image_array.shape[2],
                                  image_array.shape[0])

import scipy.misc
scipy.misc.toimage(image_array, cmin=0.0).save('masked_plot.tif')

image_plot = tiff.imread('masked_plot.tif')

ax.imshow(mark_boundaries(image_plot, segments))

plt.tight_layout()
plt.show()



