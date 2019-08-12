import numpy as np
np.random.seed(42)

import os
import geopandas as gpd
import pandas as pd
import rasterio as rio
import json
from fiona.crs import from_epsg
from shapely.geometry import box as geobox

import time
import pylab as pl

import matplotlib.pyplot as plt
from keras.models import Sequential
import land_classification as lc
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2
from keras import optimizers
from keras.utils import np_utils, to_categorical
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.applications.densenet import DenseNet201 as DenseNet
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow
# import segment as seg

root_path = os.getcwd()

print('Imports done')

# Defining Swindon area of interest

aoi_geo = geobox(-2.29, 51.51, -1.71, 51.61)
aoi = gpd.GeoDataFrame([], geometry=[aoi_geo])
aoi.crs = from_epsg(4326)
aoi.to_file('data/aoi.geojson', driver='GeoJSON')
# aoi = 'data/aoi.geojson'
# Getting land-cover classes

with open(root_path + '/data/labels.json') as jf:
    names = json.load(jf)

# Reading and merging band data

# s2_band = 'S2A.SAFE'
# data, profile = lc.merge_bands(s2_band, res='10')

# # # Writing and masking band raster

# lc.write_raster('data/swindon/merged.tif', data, profile)
# lc.mask_raster(aoi, 'data/swindon/merged.tif', 'data/swindon/masked.tif')

bands = ['B02', 'B03', 'B04', 'B08']

print('Extracting points...')

pe = lc.PointExtractor(aoi)

points_df = pe.get_n(10000)

def sample_raster(df, path, bands=['B02', 'B03', 'B04', 'B08'], buffer=1):

    assert isinstance(path, str) or isinstance(path, rio.DatasetReader)
    if isinstance(path, str):
        tif = rio.open(path)
    else:
        tif = path

    df = df.to_crs(from_epsg(tif.crs.to_epsg()))

    if tif.count == 1:
        arr = tif.read()
    else:
        arr = tif.read(list(pl.arange(tif.count) + 1))

    values = []

    for i, j in zip(*tif.index(df['geometry'].x, df['geometry'].y)):
        values.append(arr[:, i-buffer:(i+1)+buffer, j-buffer:(j+1)+buffer])

    cols = [band + '_' + str(v+1) for band in bands for v in range(values[0].shape[1] * values[0].shape[2])]
    new_df = pd.DataFrame(data=list(map(lambda x: x.flatten(), values)), columns=cols)
    df[new_df.columns] = new_df
    df = df.dropna()
    return df

points_df = sample_raster(points_df, root_path + '/data/Corine_S2_Proj_2.tif', bands=['labels'])

points_df = sample_raster(points_df, root_path + '/data/masked.tif', bands=bands)

# clean_df = lc.remove_outliers(points_df, bands=bands, indices=False)
# clean_df = lc.calc_indices(points_df)
# clean_df = clean_df.drop(['savi', 'evi', 'ndvi'], axis=1)

df = points_df

class_cols = 'labels_1'

# predictors = ['B02_1', 'B03_1', 'B04_1', 'B08_1']

df = lc.onehot_targets(df)

# cls_count = len(np.unique(y))
# y_oh = to_categorical(y, num_classes=42)
# X_train = preprocessing.scale(X_train)
# X_test = preprocessing.scale(X_test)

dfc = df.iloc[:,-17:]
dfc = dfc.rename(columns=lambda x: dfc.columns.get_loc(x))
dfd = df.iloc[:,:17]
df = pd.concat([dfd, dfc], axis=1)

X = df[predictors]
X = X.values
y = df.iloc[:,9:]
clsc = len(y.columns)
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

preds = len(predictors);preds

model = Sequential()
model.add(Dense(200, input_shape=(preds,), activation="relu"))
# model.add(Dropout(0.2))
# model.add(Dense(200, activation="relu"))
# model.add(Dropout(0.2))
# model.add(Dense(200, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(200, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(clsc, activation='softmax'))
model.summary()

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Compile model

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])

# pred, proba, cm, algo = lc.classify(clean_df, , bandsalgorithm=model)

history=model.fit(X_train,
          y_train,
          epochs=100, 
          batch_size=100, 
          validation_split = 0.2,
          verbose=1,
          )


mask_src=rio.open('data/masked.tif')
profile = mask_src.profile
data = mask_src.read(list(pl.arange(mask_src.count) + 1))
gdf = lc.create_raster_df(data, bands)
gdf = lc.calc_indices(gdf)

out=model.predict(gdf[['B02', 'B03', 'B04', 'B08']])

cls_cv = model.predict(X_test)
score = model.score(X_test, y_test)

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

# if __name__ == "__main__":
#     seg

# tif = rio.open(root_path + '/data/masked.tif')

# segment_df, segments, shapes = seg.create_segment_polygon(tif)
# seg.plot_segments(segments)
# zones_df, dist_values = seg.get_zones_and_dists(segment_df)
# zones_df = seg.tag_zones(zones_df, dist_values)
# segment_df['polygon id'] = zones_df['zone_id']
# segment_df = segment_df.dropna()
# points_df = seg.match_segment_id(points_df, segment_df)

# names2 = {v: k for k, v in names.items()}
# lbls = pd.Series(points_df.labels_1.values)

# ent_df = []

# for entry in (lbls):
#     # for elem in entry:
#     print((entry))
#     new_entry = []
#     for k,v in ldict.items():
#         for val in v:
#             if entry == val:
#                 entry = k
#     new_entry.append(entry)

# ent_df.append(new_entry)

# points_df['segment_id'] = np.nan


# def match_segment_id(points_df, segment_df):

#     st = time.time()

#     for i, row in enumerate(points_df.itertuples(), 0):
#         print(row)
#         p = points_df
#         point = points_df.at[i, 'geometry']


#         for j in range(len(segment_df)):
#               poly = segment_df.iat[j, 0]
#               print(poly)

#               if poly.contains(point):
#                   points_df.at[i, 'segment_id'] = segment_df.iat[j, 1]
#               else:
#                   pass
#     en=time.time()
#     print('pixels and segments matched in {} sec'.format(en-st))
#     return points_df


# st = time.time()

# pixels = points_df.geometry.values
# polys = segment_df

# new_ind = range(0, len(polys))
# polys = polys.reindex(new_ind)

# def aggregate_values(series, agg_dict):

#     lower_col = pd.Series(data=np.zeros(series.shape))
#     for k, v in agg_dict.items():
# #         lower_col[series.isin(v)] = k

# #     return lower_col
# for pix in points_df.geometry:
#     for poly in polys.geometry:
#         if poly.contains(pix):




# for i in range(0, len(pixels)):
#     poi = pixels[i]
#     for j in range(len(polys)):
#         poly = polys.geometry.iat[j]
#         if poly.contains(poi):
#             points_df.at[i, 'segment_id'] = segment_df.iat[j, 1]
#         else:
#             pass

# en=time.time()
# print('pixels and segments matched in {} sec'.format(en-st))

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
#
#