from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import random as rn
rn.seed(1)

import os
import time
import json
import numpy as np
import pandas as pd
import rasterio as rio
from itertools import chain
import collections
import matplotlib.pyplot as plt
import seaborn
from sklearn.utils import class_weight

# from keras.applications.xception import Xception
# from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation
from keras import optimizers
from keras.optimizers import SGD
from PIL import Image
# import cv2
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

import cv2

import read_data_df as rd

print('Imports done')

root_path = os.getcwd()

Server = False
# Server = True

cv = False
cv = True

if Server:
    path_to_images = root_path + '/DATA/bigearth/dump/sample/'
    path_to_model = root_path + '/DATA/bigearth/model/'
    with open('/home/strathclyde/DATA/corine_labels.json') as jf:
        names = json.load(jf)

else:
    path_to_images = root_path + '/data/sample/'
    path_to_model = root_path + '/data/models/'

    with open('data/corine_labels.json') as jf:
        names = json.load(jf)

st = time.time()

patches = [patches for patches in os.listdir(path_to_images)]
patches, split_point = rd.get_patches(patches)
print('patch count: {}'.format(len(patches)))
# print('split point: {}'.format(split_point))

# split_point=3751
df, class_count = rd.read_patch(split_point)
X = df.path
y = df.labels
train, test = train_test_split(df, test_size=0.2)

def fold_df(df, k):

    X = df.path
    y = df.labels
    folds = KFold(n_splits=k, shuffle=True, random_state=1).split(X, y)

    return folds, X, y

if cv:
    k = 7
    folds, X_train, y_train = fold_df(df, k)

class_rep = list(chain.from_iterable(df['labels']))
counter=collections.Counter(class_rep)
print('class dist: {}'.format(counter))
# plt.bar(range(len(counter)), list(counter.values()), align='center')
# plt.xticks(range(len(counter)), list(counter.keys()))
# plt.title('Training classes')
# plt.ion()
# plt.show()

print('class count: {}'.format(class_count))

print('df : {}'.format(df.head()))
print(set(df['path'].apply(lambda x: os.path.exists(x))))

en = time.time()
t = en - st
print('Images ready in: {} minutes'.format(int(t/60)))

def build_model():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(120, 120, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(class_count, activation='sigmoid'))

    return model

print('Building model...')

sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model = build_model()

model.compile(optimizer=sgd, loss='binary_crossentropy',
              metrics=['categorical_accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              cooldown=1,
                              patience=5,
                              min_lr=0.0001)

earlystopper = EarlyStopping(monitor='val_categorical_accuracy',
                             patience=50,
                             mode='max')



print('Normalizing images...')

def preprocessing(x):

    mean = [87.845, 96.965, 103.947]
    std = [23.657, 16.474, 13.793]

    for idx, mean_value in enumerate(mean):
        x[..., idx] -= mean_value
        x[..., idx] /= std[idx]

    return x

''' 
Server 15k sample dataset:
    
Training indices: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
training class count: Counter({2: 52888, 1: 43221, 4: 20357, 0: 11289, 3: 650})
Validation indices: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
valid class count: Counter({2: 22756, 1: 18552, 4: 8755, 0: 4790, 3: 260})
Class weights calculated below:
    [ 2.12513238  0.63897122  0.50568964 27.6042328   1.05185484] 
'''

if Server:
    classes = [i for i in range(1,6)]
else:
    classes = [i for i in range(1,6) if i != 4]   # For local sample dataset

raw_labels = df.labels.values.tolist()
raw_labels = [l for sub_l in raw_labels for l in sub_l]

class_weightings = class_weight.compute_class_weight('balanced',
                                                      classes,
                                                      raw_labels)

print('Class weights {}'.format(class_weightings))

# preprocessing_function=preprocessing,

gen = ImageDataGenerator(validation_split=0.2)

if cv:
    for j, (train_idx, val_idx) in enumerate(folds):

        # print(train_idx)
        # print(val_idx)
        print('\nFold ',j)

        X_train_cv = X_train[train_idx]
        y_train_cv = y_train[train_idx]

        dict1 = {'path': X_train_cv, 'labels': y_train_cv}
        df1 = pd.DataFrame(dict1)
        # print(df1)

        X_valid_cv = X_train[val_idx]
        y_valid_cv = y_train[val_idx]

        dict2 = {'path': X_valid_cv, 'labels': y_valid_cv}
        df2 = pd.DataFrame(dict2)
        # generator = gen.flow(X_train_cv, y_train_cv, batch_size = 64)

        generator = gen.flow_from_dataframe(
                        dataframe = train,
                        x_col = 'path',
                        y_col = 'labels',
                        seed = 1,
                        target_size = (120,120),
                        subset = 'training',
                        class_mode = 'categorical',
                        batch_size = 64,
                        shuffle = True)

        v_generator = gen.flow_from_dataframe(
                        dataframe = train,
                        x_col = 'path',
                        y_col = 'labels',
                        seed = 1,
                        target_size = (120,120),
                        subset = 'validation',
                        class_mode = 'categorical',
                        batch_size = 64,
                        shuffle = True)

        history = model.fit_generator(generator,
                            steps_per_epoch=1,
                            epochs=1,
                            validation_data=v_generator,
                            validation_steps=1,
                            class_weight = class_weightings)

        # print(model.evaluate(X_valid_cv, y_valid_cv))

        test_arrays = np.empty(len(X_valid_cv))
        predictions = []
        for i in X_valid_cv.values:
            img_ar = Image.open(i)
            img_ar = np.expand_dims(img_ar, axis=0)
            pred = model.predict(img_ar)
            pred = pred.argsort()[-2:][::-1]
            predictions.append(pred)
        # print(predictions)
        cf = confusion_matrix(test.iloc[:, 2:].values, predictions)
        print(cf)
else:

    print('Flowing training set...')

    training_data = gen.flow_from_dataframe(
                    dataframe = train,
                    x_col = 'path',
                    y_col = 'labels',
                    subset = "training",
                    seed = 1,
                    target_size = (120,120),
                    class_mode = 'categorical',
                    batch_size = 64,
                    shuffle = True)
    
    train_cls = training_data.class_indices
    train_rep = training_data.classes
    train_rep = list(chain.from_iterable(train_rep))
    counter=collections.Counter(train_rep)
    
    print('Training indices: {}'.format(train_cls))
    print('training class count: {}'.format(counter))

    print('Flowing validation set...')
    
    validation_data = gen.flow_from_dataframe(
                    dataframe = train,
                    x_col = 'path',
                    y_col = 'labels',
                    subset = "validation",
                    seed = 1,
                    target_size = (120,120),
                    class_mode = 'categorical',
                    batch_size = 64,
                    shuffle = True)
    
    valid_cls = validation_data.class_indices
    valid_rep = validation_data.classes
    valid_rep = list(chain.from_iterable(valid_rep))
    counter=collections.Counter(valid_rep)
    
    print('Validation indices: {}'.format(validation_data.class_indices))
    print('valid class count: {}'.format(counter))

    checkpointer = ModelCheckpoint(path_to_model + 'bigearth' +
                               "_rgb_" +
                               "{epoch:02d}-{val_categorical_accuracy:.3f}" +
                               ".hdf5",
                               monitor='val_categorical_accuracy',
                               verbose=1,
                               save_best_only=True,
                               mode='max')

    history = model.fit_generator(training_data,
                                    steps_per_epoch = 1,
                                    epochs = 1,
                                    validation_data = validation_data,
                                    validation_steps = 1,
                                    callbacks=[reduce_lr, earlystopper],
                                    class_weight = class_weightings)

    # test_arrays = np.empty(len(test))
    # predictions = []
    # for i in test.path.values:
    #     img_ar = cv2.imread(i)
    #     img_ar = np.expand_dims(img_ar, axis=0)
    #     pred = model.predict(img_ar)
    #     pred = pred.argsort()[-3:][::-1]
    #     print(pred)
    #     predictions.append(pred)
    # print(predictions)
    # # y_pred = model.predict(test_arrays)
    # cf = confusion_matrix(test.iloc[:, 2:].values, predictions)
    # print(cf)
    # targs = ['Artificial', 'Agriculture', 'Forest & Vegetation', 'Bare space', 'Water']
    # print(classification_report(valid_rep, y_pred, target_names=targs))


plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig('/home/strathclyde/DATA/bigearth/output/acc_15_10_0.1.jpg')
# plt.show(block=True)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('/home/strathclyde/DATA/bigearth/output/loss_15_10_0.1.jpg')
# plt.show(block=True)

# if __name__ == '__main__':
#     rd

# df['l2'] = df['labels'].apply(lambda x: str(x).strip('[]'))

# plt.bar(range(len(counter)), list(counter.values()), align='center')
# plt.xticks(range(len(counter)), list(counter.keys()))
# plt.title('Training classes')
# plt.ion()
# plt.show()
