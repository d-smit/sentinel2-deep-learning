import geopandas as gpd
import pylab as pl
from scipy import stats
import rasterio as rio
import matplotlib.pyplot as plt
import json
import pandas as pd
from geopandas import GeoDataFrame
from fiona.crs import from_epsg
from shapely.geometry import box as geobox
import land_classification as lc
from land_classification.preprocessing import create_raster_df, onehot_targets, filter_low_counts, df_pca
from land_classification.sampling import sample_raster
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

import collections

def scene_preparation():
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

    return data, profile, names, aoi

data, profile, names, aoi = scene_preparation()

pe = lc.PointExtractor(aoi)
points_df = pe.get_n(50000)
bands = ['B02', 'B03', 'B04','B08']

def create_df(df, bands):

    points_df = sample_raster(df, 'data/Corine_S2_Proj_2.tif', bands=['labels'])
    points_df = sample_raster(points_df, 'data/masked.tif', bands=bands)

    bands = ['B02_1', 'B03_1', 'B04_1','B08_1']

    return points_df, bands

df, bands = create_df(points_df, bands)

counter=collections.Counter(df.labels_1.values)

df = filter_low_counts(df, samples=500)

df = onehot_targets(df)

class_cols = list(df['labels_1'].unique())
num_class = len(class_cols)

X = df[bands]
y = df[class_cols]

X_train, y_train = X, y

def build_model():

    model = Sequential()
    model.add(Dense(200, input_shape=(4,), activation="relu"))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(num_class, activation='softmax'))
    model.summary()

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['categorical_accuracy'])

    return model

model = build_model()

history=model.fit(X_train,
          y_train,
          epochs=100, 
          batch_size=100, 
          validation_split = 0.2,
          verbose=1,
          )

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()