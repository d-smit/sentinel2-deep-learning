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
    elif isinstance(vector, gpd.GeoDataFrame):
        bounds = vector
        
    bounds = bounds.to_crs(tif.crs)
    combined = zonal_stats(vectors=bounds['geometry'],
                       raster=tif.read(1),
                       affine=tif.transform,
                       all_touched=True,
                       geojson_out=True,
                       categorical=True)
    
    df = gpd.GeoDataFrame.from_features(combined)
    df['pred_class'] = df.drop(columns='geometry').idxmax(axis=1)
    df.to_file(out_file, driver='GeoJSON')
    print('Output vector file {} written'.format(out_file))
    return df

def plot_cm(cm, plot_names, date='', name='cm.png'):
    pc_cm = pl.true_divide(cm, cm.sum(axis=1, keepdims=True))
    pc_cm = pl.nan_to_num(pc_cm)
    f, ax = pl.subplots(1, figsize = (20, 20))
    pl.set_printoptions(precision=2)
    sns.heatmap(ax = ax, 
                data=pc_cm.round(2), 
                annot=True, 
                fmt='g', 
                cmap='pink',
                linewidths=0.5, 
                cbar=False)
    ax.set_title('Confusion matrix for Corine Level-2 Groups {}'.format(date))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    pl.xticks(pl.arange(pc_cm.shape[0])+0.5, plot_names, rotation=45)
    pl.yticks(pl.arange(pc_cm.shape[1])+0.5, plot_names, rotation=45)
    f.savefig(name)