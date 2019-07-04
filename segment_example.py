import rasterio as rio
from fiona.crs import from_epsg
from shapely.geometry import box as geobox
import json
import geojson
from rasterio import features
import geopandas as gpd
# import glob
# import os
import numpy as np
from shapely.geometry import shape
import pylab as pl
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
from subprocess import check_output
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import land_classification as lc
import tifffile as tiff

img = tiff.imread('data/segment/TCI_clipped.tif')

gradient = sobel(rgb2gray(img))

segments_quick = quickshift(img, kernel_size=3, max_dist=400, ratio=0.5)
segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)


print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))
fig, ax = plt.subplots(2, 2, figsize=(25, 25), sharex=True, sharey=True)
 
ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[1, 0].imshow(mark_boundaries(img, segments_quick))
ax[1, 0].set_title('Quickshift')


for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()

print('Segments : {}'.format(segments_quick))
print('Segment shape : {}'.format(segments_quick.shape))
print('Number of segments: {}'.format(len(np.unique(segments_quick))))
print('Image shape: {}'.format(img.shape))
