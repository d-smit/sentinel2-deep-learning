from sklearn.neural_network import MLPClassifier
import rasterio as rio
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pylab as pl
import seaborn as sns
import os

from .preprocessing import create_raster_df, create_zero_samples
from .raster import calc_indices
from .io import write_raster

def classify(df, pred_path='data/masked.tif', cv=True, name='mlp', algorithm=MLPClassifier()):    
    assert isinstance(pred_path, str) or isinstance(pred_path, rio.DatasetReader)
    
    if isinstance(pred_path, str):
        mask_src = rio.open(pred_path)
    else:
        mask_src = pred_path
        
    X = df.drop(['labels'], axis=1)
    y = df[['labels']]
    
    profile = mask_src.profile
    data = mask_src.read(list(pl.arange(mask_src.count) + 1))
    gdf = create_raster_df(data)
    gdf = calc_indices(gdf)
    if cv:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    else:
        X_train, y_train = X, y
    cls = algorithm
    cls.fit(X_train, y_train['labels'].ravel())
    pred = cls.predict(gdf).reshape(1, data.shape[1], data.shape[2]).astype(pl.int16)
    proba = cls.predict_proba(gdf).max(axis=1).reshape(1, data.shape[1], data.shape[2])
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    write_raster("outputs/lc_10m_{}_pred.tif".format(name), pred, profile, nodata=0)
    write_raster("outputs/lc_10m_{}_proba.tif".format(name), proba, profile)
    
    if cv:
        cls_cv = cls.predict(X_test)
        score = cls.score(X_test, y_test)
        print(score)
        cm = confusion_matrix(cls_cv, y_test)
        f, ax = pl.subplots(1, figsize = (20, 20))
        sns.heatmap(ax = ax, data=cm, annot=True, fmt='g', linewidths=0.5, cbar=False)
        f.savefig('outputs/cv_{}.png'.format(name))
    return pred, proba, cm