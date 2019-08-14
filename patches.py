import numpy as np
import tensorflow as tf
SEED = 12345
tf.set_random_seed(SEED)
np.random.seed(SEED)

import os
import geopandas as gpd
from scipy import stats
import matplotlib.pyplot as plt
import json
import pylab as pl
import pandas as pd
from fiona.crs import from_epsg
from shapely.geometry import box as geobox
import land_classification as lc
from land_classification.preprocessing import create_raster_df, remove_outliers, onehot_targets, filter_low_counts, df_pca
from land_classification.sampling import sample_raster
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
import collections
import segment as seg
from time import time
from skimage.io import imread
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from sklearn.utils import class_weight

root_path = os.getcwd()

Segment = False
# Server = True

path_to_model = root_path + '/models/'

# def prepare_scene():
#     aoi_geo = geobox(-2.29, 51.51, -1.71, 51.61)
#     aoi = gpd.GeoDataFrame([], geometry=[aoi_geo])
#     aoi.crs = from_epsg(4326)
#     aoi.to_file('data/aoi.geojson', driver='GeoJSON')
    
#     with open('data/labels.json') as jf:
#         names = json.load(jf)
#     s2_band = 'S2A.SAFE'

#     data, profile = lc.merge_bands(s2_band, res='10')
    
#     lc.write_raster('data/merged.tif', data, profile)
#     lc.mask_raster(aoi, 'data/merged.tif', 'data/masked.tif')

#     return data, profile, names, aoi

# data, profile, names, aoi = prepare_scene()
# # tif = rio.open('data/masked.tif', 'r')

# def segment_scene(tif):
#     seg_df, segments, shapes = seg.create_segment_polygon(tif)
#     seg.plot_segments(segments)
#     zones_df, dists = seg.get_zones_and_dists(seg_df)
#     zones_df = seg.tag_zones(zones_df, dists)
#     seg_df['zone_id'] = zones_df['zone_id']

#     return seg_df

# # seg_df = segment_scene(tif)

# pe = lc.PointExtractor(aoi)
# points_df = pe.get_n(5000)

# def create_df(df, bands= ['B02', 'B03', 'B04', 'B08']):

#     points_df, values = sample_raster(df, 'data/Corine_S2_Proj_2.tif', bands=['labels'])

#     if Segment:
#         points_df = seg.match_segment_id(points_df, seg_df)
#         points_df = sample_raster(points_df, 'data/masked.tif', bands=bands)
#     else:
#         points_df, values = sample_raster(points_df, 'data/masked.tif', bands=bands)

#     # bands = ['B02_1', 'B03_1', 'B04_1','B08_1']
#     points_df.to_file('points_shp_ex.geojson', driver='GeoJSON')
#     np.savez_compressed('patch_arrays.npz', values)

#     print('pixel df stored')

#     return points_df, values

# df, values = create_df(points_df)

def check_sim(df):
    for idx, row in df[label_cols].iterrows():
        sim = [i for i in row if i != row[0]]
        if (len(sim)) > int(len(row)/4):
            df.drop(idx, inplace=True)
    return df

print('reading points df')
df = gpd.read_file('points_shp_ex.geojson')
values = np.load('patch_arrays.npz')
patches = values['arr_0']
patches = patches.astype(float)
patches = patches[:len(df)]

def sort_cols(df):
    df = df.drop(['Lat', 'Lon', 'Val', 'geometry'], axis=1)
    df = lc.calc_indices(df)

    label_cols = df.columns[pd.Series(df.columns).str.startswith('lab', na=False)]

    band_cols = df.columns[pd.Series(df.columns).str.startswith('B', na=False)]
    indices = df.columns[pd.Series(df.columns).str.startswith('nd', na=False)]

    pred_cols = band_cols.to_list() + indices.to_list()

    labels = label_cols[1:]
    df = df.drop(labels, axis=1)

    return df

df = sort_cols(df)

counter = collections.Counter(df.labels_1.values)
print('class dist: {}'.format(counter))

def clean_scale(df, patches):

    # df = df[(np.abs(stats.zscore(df)) < 1.5).all(axis=1)]
    # df = filter_low_counts(df, samples=100)
    df = onehot_targets(df)
    print('loading patch arrays and scaling...')

    arrays = []
    scalers = {}
    for patch in patches:
        patch = patch.astype(float)
        for i in range(patch.shape[0]):
            scalers[i] = StandardScaler()
            patch[i,:,:] = scalers[i].fit_transform(patch[i,:,:])
        patch = np.moveaxis(patch, 0, 2)
        arrays.append(patch)

    return df, arrays

df, arrays = clean_scale(df, patches)

X = np.asarray(arrays)

class_cols = list(df['labels_1'].unique())
y = df[class_cols]

X_train, y_train = X, y

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def build_model(in_shape=(X[1].shape), out_shape=len(y.columns)):
     model = Sequential()
     model.add(Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=in_shape))
     model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
     # model.add(MaxPooling2D((2, 2)))
     # model.add(Dropout(0.2))
     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
     model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
      # model.add(MaxPooling2D((2, 2)))
      # model.add(Dropout(0.2))
     model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
     # model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
     # # model.add(MaxPooling2D((2, 2)))
     # model.add(Dropout(0.2))
     model.add(Flatten())
     model.add(Dense(3200, activation='relu', kernel_initializer='he_uniform'))
     # model.add(Dropout(0.5))
     model.add(Dense(out_shape, activation='softmax'))
     adam = optimizers.Adam(lr=0.00075)

     model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
     return model

model = build_model()
class_weightings = class_weight.compute_class_weight('balanced',
                                                      df.labels_1.unique(),
                                                      df.labels_1.values)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              cooldown=1,
                              patience=5,
                              min_lr=0.0000001)

earlystopper = EarlyStopping(monitor='val_acc',
                              patience=10,
                              mode='max')

checkpointer = ModelCheckpoint(path_to_model + 'CNNPatch' +
                                "_rgb_" +
                                "{epoch:02d}-{val_acc:.3f}" +
                                ".hdf5",
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=False,
                                mode='max')

history=model.fit(X_train,
          y_train,
          epochs=250,
          batch_size=256,
          validation_split = 0.1,
          verbose=1,
          callbacks=[reduce_lr, checkpointer, earlystopper],
          class_weight=class_weightings)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

if Server:
    plt.savefig('/home/strathclyde/seg/plots/acc_11x11.jpg')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

if Server:
    plt.savefig('/home/strathclyde/seg/plots/loss_11x11.jpg')

# pred, proba, cm, algo = lc.classify(df, onehot=True)

# def build_model():

#     model = Sequential()
#     model.add(Dense(400, input_shape=(len(X.columns),), activation="relu"))
#     # model.add(Dropout(rate=0.2))
#     model.add(Dense(300, kernel_regularizer=l2(l=0.01), activation="relu"))
#     # model.add(Dropout(rate=0.4))

#     model.add(Dense(200, activation='relu'))
#     # model.add(Dropout(rate=0.2))
#     model.add(Dense(100, activation='relu'))
#     # model.add(Dropout(rate=0.2))

#     model.add(Dense(len(y.columns), activation='softmax'))
#     model.summary()

    # sgd = optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True)
    # adam = optimizers.Adam(lr=0.0075)#, decay=1e-6, momentum=0.9, nesterov=True)

#     model.compile(loss='categorical_crossentropy',
#                   optimizer=adam,
#                   metrics=['acc'])

#     return model


# # def match_id(pixel):
# #     for x in range(len(seg_df)):
# #         if seg_df['geometry'][x].contains(pixel):
# #             zone = seg_df['zone_id'][x]
# #     return zone

# # st = time()

# # points_df['zone_id'] = points_df['geometry'].apply(lambda x: match_id(x))
# # en=time()
# print('pixels and segments matched in {} sec'.format(en-st))
# match_id(pixel)

# pixel = points_df['geometry'][1]
# print(match_id(pixel))
# seg_df.info()
# seg_df['zone_id']
# d=seg_df['zone_id'].where(seg_df['geometry'].apply(lambda x : x.contains(pixel)))
# d.count()
# seg_df['geometry'].apply(lambda x: x.contains(pixel))

# scalers = {}
# for i in range(X.shape[2]):
#     scalers[i] = StandardScaler()
#     X[:, i, :] = scalers[i].fit_transform(X[:, i, :])
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(X)
# X = pd.DataFrame(scaled_data)
# X_train = preprocessing.scale(X_train)
# X_test = preprocessing.scale(X_test)
