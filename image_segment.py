import rasterio as rio
import geopandas as gpd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from fiona.crs import from_epsg
from shapely.geometry import box as geobox
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import land_classification as lc

#tif = rio.open('data/swindon/masked.tif')
#
#data = pl.stack(list((lambda x: x.read(1).astype(pl.int16), tif)))
#
#s2_band = 'S2A.SAFE'
#
#data, profile = lc.merge_bands(s2_band, res='10')
#
#tif = data[0:3]
#
#tif = tif.reshape(tif.shape[1], tif.shape[2], tif.shape[0])

img = img_as_float(astronaut()[::2, ::2])

segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
gradient = sobel(rgb2gray(img))
segments_watershed = watershed(gradient, markers=250, compactness=0.001)

print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img, segments_quick))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()
#tif = rio.open('swindon_TCI.jp2')
#
#
aoi_geo = geobox(-2.29, 51.51, -1.71, 51.61)
aoi = gpd.GeoDataFrame([], geometry=[aoi_geo])
aoi.crs = from_epsg(4326)
aoi.to_file('data/aoi.geojson', driver='GeoJSON')

s2_band = 'S2A.SAFE'

data, profile = lc.merge_bands(s2_band, res='10')
lc.write_raster('data/segment/merged.tif', data, profile)
lc.mask_raster(aoi, 'data/segment/merged.tif', 'data/segment/masked.tif')
#tif = data[0:3]
#
#data = data.reshape(data.shape[1], data.shape[2], data.shape[0])

#img = img_as_float(astronaut()[::2, ::2])


segments_quick = quickshift(data.astype(np.int32), kernel_size=3, max_dist=6, ratio=0.5)
#gradient = sobel(rgb2gray(img))
#segments_watershed = watershed(gradient, markers=250, compactness=0.001)

#print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
#print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
#print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))
#
#fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
#
#ax[0, 0].imshow(mark_boundaries(img, segments_fz))
#ax[0, 0].set_title("Felzenszwalbs's method")
#ax[0, 1].imshow(mark_boundaries(img, segments_slic))
#ax[0, 1].set_title('SLIC')
#ax[1, 0].imshow(mark_boundaries(img, segments_quick))
#ax[1, 0].set_title('Quickshift')
#ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
#ax[1, 1].set_title('Compact watershed')
#
#for a in ax.ravel():
#    a.set_axis_off()
#
#plt.tight_layout()
#plt.show()

