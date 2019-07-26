from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import random as rn
rn.seed(1)

import os
import pandas as pd
import numpy as np
import pylab as pl
import json
import matplotlib.pyplot as plt

from keras.applications.xception import Xception
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.backend import tensorflow_backend
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

import read_data_df as rd

print('Imports done')

root_path = os.getcwd()

path_to_images = root_path + '/data/merge/'
path_to_split = root_path + '/data/split/'
path_to_train = root_path  + '/data/split/train/'
path_to_validation = root_path + '/data/split/validation/'
path_to_merge = root_path + '/data/merge/'

with open('data/corine_labels.json') as jf:
    names = json.load(jf)
rd.build_dirs()
patches = [patches for patches in os.listdir(path_to_images)]

patches, split_point = rd.get_patches(patches)

class_count, df = rd.read_patch(split_point)

X = df.path
y = df.labels

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

train_dict = {'path': X_train, 'labels': y_train}
train_df = pd.DataFrame(train_dict)

test_dict = {'path': X_test, 'labels': y_test}
test_df = pd.DataFrame(test_dict)

num_classes = len(names.values())

# path_check_shape = path_to_train + 'Coniferous forest/'
# tifs = [tif for tif in os.listdir(path_check_shape)]
# im = Image.open(path_check_shape+tifs[10])
# im = np.array(im)
# input_shape = im.shape

# Basic CNN

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(120,120,3)))
model.add(BatchNormalization())
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# ConvNet

# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# Xception

# base_model = Xception(input_shape=(256, 256, 3), weights=None)
# separate model for giving the class predictions
# top_model = base_model.output
# need pooling in between top and base models?
# preds = Dense(num_classes, activation='softmax')(top_model)
# #compile into single model object with Model class
# model = Model(inputs=base_model.input, outputs=preds)

print('Normalizing images...')

train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

print('Flowing training set...')

training_data = train_datagen.flow_from_dataframe(
                dataframe = train_df,
                x_col = 'path',
                y_col = 'labels',
                subset = "training",
                seed = 1,
                target_size = (120,120),
                classes = names.values(),
                class_mode = 'categorical',
                batch_size = 64,
                shuffle = True)

print('Flowing validation set...')

testing_data = test_datagen.flow_from_dataframe(
                dataframe = test_df,
                x_col = 'path',
                y_col = 'labels',
                subset = "validation",
                seed = 1,
                target_size = (120,120),
                classes = names.values(),
                class_mode = 'categorical',
                batch_size = 64,
                shuffle = True)

print('Training network...')

model.compile(optimizer='adadelta', loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

history = model.fit_generator(training_data,
                    steps_per_epoch=1000,
                    epochs=5,
                    validation_data=testing_data,
                    validation_steps=500)

keys = history.history.keys

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show(block=True)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show(block=True)

if __name__ == "__main__":
    rd
    rd.build_dirs()
    rd.get_patches()
    rd.read_patch()