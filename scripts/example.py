import geopandas as gpd
import json
from fiona.crs import from_epsg
from shapely.geometry import box as geobox
import land_classification as lc


aoi_geo = geobox(-2.29, 51.51, -1.71, 51.61)
aoi = gpd.GeoDataFrame([], geometry=[aoi_geo])
aoi.crs = from_epsg(4326)
aoi.to_file('data/aoi.geojson', driver='GeoJSON')

with open('data/labels.json') as jf:
    names = json.load(jf)
s2_band = 'S2A.SAFE'

data, profile = lc.merge_bands(s2_band, res='10')

lc.write_raster('data/merged.tif', data, profile)
lc.mask_raster(aoi, 'data/merged.tif', 'data/masked.tif')

pe = lc.PointExtractor(aoi)
points_df = pe.get_n(250)
bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08']

points_df = lc.sample_raster(points_df, 'data/Corine_S2_Proj_2.tif', bands=['labels'])
points_df = lc.sample_raster(points_df, 'data/masked.tif', bands=bands)

clean_df = lc.remove_outliers(points_df, bands=bands, indices=False)
clean_df = lc.calc_indices(clean_df)

pred, proba, cm, cls = lc.classify(clean_df, onehot=False, labels=names)

print('Predictions: \n{}'.format(pred))
print('Probabilities: \n{}'.format(proba))
print('Confusion Matrix: \n{}'.format(cm.shape))
print('Classifier parameters: \n{}'.format(cls))


