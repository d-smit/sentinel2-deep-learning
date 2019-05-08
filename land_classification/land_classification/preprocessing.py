import pylab as pl
from pandas import concat
from scipy import stats
from geopandas import GeoDataFrame
from .raster import calc_indices

def remove_outliers(df, bands=['B02', 'B03', 'B04', 'B08'], indices=False):
    lab_grp = df.groupby('labels')
    lab_dfs = []
    
    for i in lab_grp.groups:
        lab_df = lab_grp.get_group(i)[bands]
        lab_clean = lab_df[(pl.absolute(stats.zscore(lab_df)) < 1).all(axis=1)]
        lab_clean['labels'] = i
        lab_dfs.append(lab_clean)
        
    clean_df = concat(lab_dfs).reset_index(drop=True)
    
    if indices:
        clean_df = calc_indices(clean_df)
        
    return clean_df


def create_raster_df(pred_array, bands=['B02', 'B03', 'B04', 'B08'], indices=False):
    X_pred = pred_array.reshape(pred_array.shape[0],
                                pl.array(pred_array.shape[1:]).prod()).T
    gdf = GeoDataFrame(X_pred, columns = bands)
    if indices:
        gdf = calc_indices(gdf)
    return gdf

def create_zero_samples(df, bands=['B02', 'B03', 'B04', 'B08', 'labels']):
    label_col = [0]
    na_df = GeoDataFrame(pl.ones((1000, len(df.columns))) * ([-9999, -9999, -9999, -9999] + label_col), columns = list(df.columns))
    df = df.append(na_df).astype(pl.int16)
    return df