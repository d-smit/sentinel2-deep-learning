import numpy as np
import tensorflow as tf
SEED = 12345
tf.set_random_seed(SEED)
np.random.seed(SEED)

import os
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras import optimizers
from sklearn.preprocessing import StandardScaler
import collections
# import segment as seg
from time import time
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from sklearn.utils import class_weight

root_path = os.getcwd()

Segment = False
Server = True
# Server = False

path_to_model = root_path + '/models/'

print('reading points df')



df = pd.read_csv('points.csv')

values = np.load('patch_arrays.npz')
patches = values['arr_0']
patches = patches.astype(float)
patches = patches[:len(df)]

def sort_cols(df):
    df = df.drop([df.columns[0]], axis='columns')
    df = df.drop(['Lat', 'Lon', 'Val', 'geometry'], axis=1)
    # df = calc_indices(df)

    label_cols = df.columns[pd.Series(df.columns).str.startswith('lab', na=False)]

    band_cols = df.columns[pd.Series(df.columns).str.startswith('B', na=False)]
    indices = df.columns[pd.Series(df.columns).str.startswith('nd', na=False)]

    pred_cols = band_cols.to_list() + indices.to_list()

    labels = label_cols[1:]
    df = df.drop(labels, axis=1)

    return df, label_cols

df, label_cols = sort_cols(df)

def check_sim(df):
    for idx, row in df[label_cols].iterrows():
        sim = [i for i in row if i != row[0]]
        if (len(sim)) > int(len(row)/4):
            df.drop(idx, inplace=True)
    return df

counter = collections.Counter(df.labels_1.values)
print('class dist: {}'.format(counter))

def clean_scale(df, patches):


    df.groupby('labels_1').filter(lambda x: x.shape[0] > 100)
    onehot = pd.get_dummies(df['labels_1'])
    df[onehot.columns] = onehot
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

    np.savez_compressed('scl_arrays.npz', arrays)
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
     model.add(Dropout(0.2))
     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
     model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
     # model.add(MaxPooling2D((2, 2)))
     model.add(Dropout(0.2))
     model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
     # model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
     # # model.add(MaxPooling2D((2, 2)))
     model.add(Dropout(0.2))
     model.add(Flatten())
     # model.add(Dense(1200, activation='relu', kernel_initializer='he_uniform'))
     model.add(Dense(2500, activation='relu', kernel_initializer='he_uniform'))
     model.add(Dropout(0.2))
     model.add(Dense(out_shape, activation='softmax'))
     adam = optimizers.Adam(lr=0.00001)

     model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
     return model

model = build_model()

class_weightings = class_weight.compute_class_weight('balanced',
                                                      df.labels_1.unique(),
                                                      df.labels_1.values)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              cooldown=1,
                              patience=10,
                              min_lr=0.000001)

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
          batch_size=512,
          validation_split = 0.3,
          verbose=1,
          callbacks=[reduce_lr, checkpointer, earlystopper],
          class_weight=class_weightings)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

if Server:
    plt.savefig('/home/strathclyde/seg/plots/acc_11x11.jpg')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

if Server:
    plt.savefig('/home/strathclyde/seg/plots/loss_11x11.jpg')

# pred, proba, cm, algo = lc.classify(df, onehot=True)

# def build_model():
    # df = df[(np.abs(stats.zscore(df)) < 1.5).all(axis=1)]
    # df = filter_low_counts(df, samples=100)
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
# df = json.read_file('points_shp_exj.json')
# with open('points_shp_ex.geojson') as f:
#     data = json.load(f)

# df = gpd.read_file('points_shp_ex.geojson')
# import json
# import pylab as pl
# from fiona.crs import from_epsg
# from shapely.geometry import box as geobox
# import land_classification as lc
# from land_classification.preprocessing import onehot_targets
# from land_classification.raster import calc_indices
# import json
# import pylab as pl
# from fiona.crs import from_epsg
# from shapely.geometry import box as geobox
# import land_classification as lc
# from land_classification.preprocessing import onehot_targets
# from land_classification.raster import calc_indices