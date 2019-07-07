import geopandas as gpd
import rasterio as rio
from rasterio import features
from rasterio.merge import merge
from rasterio.plot import show
import pylab as pl
from fiona.crs import from_epsg
from shapely.geometry import box as geobox
from shapely.geometry import shape
import json
import geojson
import numpy as np
import pandas as pd
import os
from glob import glob
from subprocess import check_output
import matplotlib.pyplot as plt
from keras.models import Sequential
import land_classification as lc
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from skimage.segmentation import felzenszwalb, quickshift

from segment import get_zones_and_dists
# Defining Swindon area of interest
print('import done')

# aoi_geo = geobox(-2.29, 51.51, -1.71, 51.61)
# aoi = gpd.GeoDataFrame([], geometry=[aoi_geo])
# aoi.crs = from_epsg(4326)
# aoi.to_file('data/aoi.geojson', driver='GeoJSON')

# # Getting land-cover classes 

# with open('data/labels.json') as jf:
#     names = json.load(jf)
    
# root_path = check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode()


# # Reading and merging band data

# s2_band = 'S2A.SAFE'
# data, profile = lc.merge_bands(s2_band, res='10')

# # Writing and masking band raster

# lc.write_raster('data/swindon/merged.tif', data, profile)
# lc.mask_raster(aoi, 'data/swindon/merged.tif', 'data/swindon/masked.tif')

pe = lc.PointExtractor(aoi)
 
points_df = pe.get_n(500)

bands = ['B02', 'B03', 'B04', 'B08']

points_df = lc.sample_raster(points_df, 'data/Corine_S2_Proj_2.tif', bands=['labels'])
points_df = lc.sample_raster(points_df, 'data/swindon/masked.tif', bands=bands)
 
clean_df = lc.remove_outliers(points_df, bands=bands, indices=False)
clean_df = lc.calc_indices(clean_df)

class_cols = 'labels_1'
 
predictors = ['B02_1', 'B03_1', 'B04_1', 'B08_1', 'ndwi']

clean_df = clean_df.drop(['savi'], axis=1)
clean_df = clean_df.drop(['evi'], axis=1)
clean_df = clean_df.drop(['ndvi'], axis=1)

X = clean_df[predictors]
X = X.values
y = clean_df[class_cols]
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

preds = len(predictors);preds
labs = len(list(clean_df[class_cols].unique()))

input_num_units = preds
hidden1_num_units = 200
hidden2_num_units = 200
hidden3_num_units = 200
hidden4_num_units = 200
output_num_units = labs

model = Sequential([
    Dense(output_dim=hidden1_num_units,
          input_dim=input_num_units,
          kernel_regularizer=l2(0.0001),
          activation='relu'),
    Dropout(0.2),
    Dense(output_dim=hidden2_num_units,
          input_dim=hidden1_num_units,
          kernel_regularizer=l2(0.0001),
          activation='relu'),
    Dropout(0.2),
    Dense(output_dim=hidden3_num_units,
          input_dim=hidden2_num_units,
          kernel_regularizer=l2(0.0001),
          activation='relu'),
    Dropout(0.1),
    Dense(output_dim=hidden4_num_units,
          input_dim=hidden3_num_units,
          kernel_regularizer=l2(0.0001),
          activation='relu'),
    Dropout(0.1),
    Dense(output_dim=(max(clean_df[class_cols])+1),
          input_dim=hidden4_num_units, 
          activation='softmax'),
 ])
    
model.summary()

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Compile model

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

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

# Model evaluation with test data set 
# Prediction at test data set

y_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test, batch_size=100, verbose=1)
print(score)
print("Baseline Error: %.2f%%" % (100-score[1]*100))

if __name__ == "__main__":
    get_zones_and_dists