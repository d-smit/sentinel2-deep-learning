import geopandas as gpd
import rasterio as rio
from fiona.crs import from_epsg
from shapely.geometry import box as geobox
import land_classification.land_classification as lc

aoi_geo = geobox(-2.29, 51.51, -1.71, 51.61)
aoi = gpd.GeoDataFrame([], geometry=[aoi_geo])
aoi.crs = from_epsg(4326)
aoi.to_file('data/aoi.geojson', driver='GeoJSON')


s2_band = 'S2A.SAFE'
data, profile = lc.merge_bands(s2_band, res='10')
# print(profile)
#
lc.write_raster('data/merged.tif', data, profile)
#
lc.mask_raster(aoi, 'data/merged.tif', 'data/masked.tif')

lc.copy_projection('data/masked.tif', 'data/Corine_Orig_10m_OS_AoI1.tif', 'data/Corine_S2_Proj_2.tif')

mask_src = rio.open('data/masked.tif')
profile = mask_src.profile
print(mask_src)
