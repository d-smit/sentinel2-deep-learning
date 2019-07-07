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

import segment as seg

print('Imports done')

# Defining Swindon area of interest

aoi_geo = geobox(-2.29, 51.51, -1.71, 51.61)
aoi = gpd.GeoDataFrame([], geometry=[aoi_geo])
aoi.crs = from_epsg(4326)
aoi.to_file('data/aoi.geojson', driver='GeoJSON')

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

tif = rio.open('data/swindon/masked.tif')

segment_df, segments, shapes = seg.create_segment_polygon(tif)

zones_df, dist_values = seg.get_zones_and_dists(segment_df)

zones_df = seg.tag_zones(zones_df, dist_values)

segment_df['polygon id'] = zones_df['zone_id']

segment_df = segment_df.dropna()

pe = lc.PointExtractor(aoi)

points_df = pe.get_n(50000)

bands = ['B02', 'B03', 'B04', 'B08']

points_df = lc.sample_raster(points_df, 'data/Corine_S2_Proj_2.tif', bands=['labels'])

points_df['segment_id'] = np.nan

# points_test = points_df.iloc[0:1000,:]
# segment_test = segment_df.iloc[0:1000,:]
#points_df['segment_id'] = points_df.iloc[:,-1].apply(lambda x: )
#  zones_df['zone_id'] = zones_df['zone_id'].apply(lambda row: mean_comp(dv, row))
#  points_df = match_segment_id(points_test, segment_test)

# def match_segment_id(pixel_df, poly_df):
#     print('Matching pixels with their segment ID...')

#     ''' Parsing through the extracted points and matching
#         each pixel with the segment ID of the segment
#         that contains it. '''

#     #for i, row in enumerate(pixel_df.itertuples(), 0):

#     def comp(pixel_df, poly_df):
#         point = pixel_df.at[x, 'geometry']

#         for j in range(len(poly_df)):
#               poly = poly_df.iat[j, 0]

#               if poly.contains(point):
#                   pixel_df.at[x, 'segment_id'] = poly_df.iat[j, 1]
#               else:
#                   pass

#     pixel_df.apply(lambda x: comp(pixel_df, poly_df, x), axis=1)

#     return pixel_df

# def comp(pixel_df, poly_df, row):
#     point = pixel_df['geometry'][ind]

#     for j in range(len(poly_df)):
#           poly = poly_df.iat[j, 0]

#           if poly.contains(point):
#               pixel_df.at[row, 'segment_id'] = poly_df.iat[j, 1]
#           else:
#               pass

# points_df[3].apply(lambda row: comp(points_df, segment_df, row), axis=1)

points_df = seg.match_segment_id(points_df, segment_df)

points_df = lc.sample_raster(points_df, 'data/swindon/masked.tif', bands=bands)
 
clean_df = lc.remove_outliers(points_df, bands=bands, indices=False)
clean_df = lc.calc_indices(clean_df)
clean_df = clean_df.drop(['savi', 'evi', 'ndvi'], axis=1)

class_cols = 'labels_1'
 
predictors = ['B02_1', 'B03_1', 'B04_1', 'B08_1', 'ndwi', 'segment_id']

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
    seg