"""
Dataframe operations on the dataset after extracting values from rasters.
"""

import pylab as pl
from pandas import concat
from scipy import stats
from sklearn.decomposition import PCA
from geopandas import GeoDataFrame
from pandas import concat, get_dummies


def remove_outliers(df, bands=['B02', 'B03', 'B04', 'B08'], indices=False):
    """
    Removes outliers from a dataframe by grouping them and deleting everything that 
    falls outside one standard deviation in the distribution of values.
    """
    lab_grp = df.groupby('labels')
    lab_dfs = []
    
    for i in lab_grp.groups:
        lab_df = lab_grp.get_group(i)[bands]
        lab_clean = lab_df[(pl.absolute(stats.zscore(lab_df)) < 1).all(axis=1)]
        lab_clean['labels'] = i
        lab_dfs.append(lab_clean)
        
    clean_df = concat(lab_dfs).reset_index(drop=True)
        
    return clean_df

def create_raster_df(pred_array, bands=['B02', 'B03', 'B04', 'B08'], indices=False):
    """
    Generates a dataframe from a raster. Each pixel becomes a row.
    """
    X_pred = pred_array.reshape(pred_array.shape[0],
                                pl.array(pred_array.shape[1:]).prod()).T
    gdf = GeoDataFrame(X_pred, columns = bands)
    if indices:
        gdf = calc_indices(gdf)
    return gdf

def create_zero_samples(df, bands=['B02', 'B03', 'B04', 'B08', 'labels'], samples=1000):
    """
    Creates a set of NaN values. Required if you have NaN values throwing off your predictions.
    
    You may try for yourself whether or not you need this. It *can* overfit on these.
    """
    label_col = [0]
    na_df = GeoDataFrame(pl.ones((samples, len(df.columns))) * ([-9999, -9999, -9999, -9999] + label_col), columns = list(df.columns))
    df = df.append(na_df).astype(pl.int16)
    return df

def balance_samples(df, samples=1000):
    """
    Balances all classes in the dataset to improve regularisation. 
    Necessary if you're using a neural network that doesn't regularise itself.
    """
    return df.groupby('labels').apply(lambda x: x.sample(samples)).reset_index(drop=True)

def filter_low_counts(df, samples=1000):
    """
    Remove classes with fewer than N samples.
    """
    return df.groupby('labels').filter(lambda x: x.shape[0] > samples)

def onehot_targets(df, column='labels'):
    """
    Converts targets to a one-hot set of arrays. Neural networks love this.
    
    i.e.
    [1 2 0 3 4 3 0] becomes
    
    [0 1 0 0 0]
    [0 0 1 0 0]
    [1 0 0 0 0]
    [0 0 0 1 0]
    [0 0 0 0 1]
    [0 0 0 1 0]
    [1 0 0 0 0]
    
    make sure that your input dataframe has a 'labels' column.
    
    """
    onehot = get_dummies(clean_df[column])
    df[onehot.columns] = onehot
    return df

def df_pca(df):
    """
    Clean imagery based on Principal Components Analysis.
    """
    grp = df.groupby('labels')
    pca = PCA(n_components=2)
    cleaned_list = []
    for ix, gp in grp:
        if ix == 0:
            cleaned_list.append(gp)
        clean_gp = gp[~pl.any((abs(pca.fit_transform(gp)) < 250) == False, axis=1)]
        cleaned_list.append(clean_gp)
    samp_df = pd.concat(cleaned_list)
    return samp_df