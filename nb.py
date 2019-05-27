import geopandas as gpd
import pylab as pl
import rasterio as rio
import json
from geopandas import GeoDataFrame
from fiona.crs import from_epsg
from shapely.geometry import box as geobox
from scipy import stats

import land_classification.land_classification as lc

aoi_geo = geobox(-2.29, 51.51, -1.71, 51.61)
aoi = gpd.GeoDataFrame([], geometry=[aoi_geo])
aoi.crs = from_epsg(4326)
aoi.to_file('data/aoi.geojson', driver='GeoJSON')

with open('data/labels.json') as jf:
    names = json.load(jf)

s2_band = 'S2A.SAFE'

# Combining each band into one multi-dimensional numpy array of uniform resolution

data, profile = lc.merge_bands(s2_band, res='10')

# Gives shape (8, 10980, 10980) where each "level" is an array representing the pixel values of that band

lc.write_raster('data/merged.tif', data, profile)
lc.mask_raster(aoi, 'data/merged.tif', 'data/masked.tif')

pe = lc.PointExtractor(aoi)
points_df = pe.get_n(1000)

bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08']

# points_df = lc.sample_raster(points_df, 'data/Corine_Orig_10m_OS_AoI1.tif', bands=['labels'])
points_df = lc.sample_raster(points_df, 'data/Corine_S2_Proj_2.tif', bands=['labels'])
print(points_df)

''' Sample raster function reads the extracted points and gets the corresponding 
Corine pixel value, labelling the dataframe. '''

points_df = lc.sample_raster(points_df, 'data/masked.tif', bands=bands)

# Used again here to get each extracted points corresponding band values.

clean_df = lc.remove_outliers(points_df, bands=bands, indices=False)
# clean_df = lc.create_zero_samples(clean_df)
clean_df = lc.calc_indices(clean_df)

pred, proba, cm, cls = lc.classify(clean_df, onehot=False, labels=names)

print('Predictions: \n{}'.format(pred))
print('Probabilities: \n{}'.format(proba))
print('Confusion Matrix: \n{}'.format(cm.shape))
print('Classifier parameters: \n{}'.format(cls))

mask_src = rio.open('data/masked.tif')

# Within .ml classify function

assert isinstance('data/masked.tif', str) or isinstance('data/masked.tif', rio.DatasetReader)
if isinstance('data/masked.tif', str):
    mask_src = rio.open('data/masked.tif')
else:
    mask_src = 'data/masked.tif'

class_cols = 'labels'

X = clean_df.drop(class_cols, axis=1)

y = clean_df[class_cols];print(set(y))

profile = mask_src.profile
data = mask_src.read(list(pl.arange(mask_src.count) + 1))

# Create test dataset

X_pred = data.reshape(data.shape[0], pl.array(data.shape[1:]).prod()).T

gdf = GeoDataFrame(X_pred, columns=bands)

