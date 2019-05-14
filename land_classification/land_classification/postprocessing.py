from rasterstats import zonal_stats
import seaborn as sns
import pylab as pl
import geopandas as gpd

def aggregate_raster(raster, vector, out_file='grouped.json'):
    """
    Aggregates raster pixels to polygons in a shapefile if you have those available. 
    Useful for post processing.
    """
    out_file = 'outputs/' + out_file
    if isinstance(raster, str):
        tif = rio.open(raster)
    elif isinstance(raster, rio.DatasetReader):
        tif = raster
        
    if isinstance(vector, str):
        bounds = gpd.read_file(vector)
    elif isinstance(bounds, gpd.GeoDataFrame):
        bounds = vector
        
    bounds = bounds.to_crs(tif.crs)
    combined = zonal_stats(vectors=bounds['geometry'],
                       raster=tif.read(),
                       affine=tif.transform,
                       all_touched=True,
                       geojson_out=True,
                       categorical=True)
    
    df = gpd.GeoDataFrame.from_features(combined)
    df['pred_class'] = df.drop(columns='geometry').idxmax(axis=1)
    df.to_file(out_file, driver='GeoJSON')
    print('Output vector file {} written'.format(out_file))
    return df