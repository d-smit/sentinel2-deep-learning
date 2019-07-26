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
import matplotlib.pyplot as plt

# from keras.applications.xception import Xception
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
# from keras.optimizers import SGD
from PIL import Image
import tensorflow as tf
from keras.backend import tensorflow_backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

import read_data_df as rd

print('Imports done')

root_path = os.getcwd()

Server = True

if Server:
    with open('/home/strathclyde/DATA/corine_labels.json') as jf:
        names = json.load(jf)

    path_to_images = root_path + '/DATA/bigearth/sample/'
    path_to_merge = root_path + '/DATA/bigearth/merge/'

    print('image store: {}'.format(path_to_images))
    print('merged to use: {}'.format(path_to_merge))

else:
    with open('data/corine_labels.json') as jf:
        names = json.load(jf)

    path_to_images = root_path + '/data/sample/'
    path_to_merge = root_path + '/data/merge/'

st = time.time()

patches = [patches for patches in os.listdir(path_to_images)]

patches, split_point = rd.get_patches(patches)

print('patch count: {}'.format(len(patches)))
print('split point: {}'.format(split_point))

class_count, df = rd.read_patch(split_point)

# print('df : {}'.format(df.head()))
# print(set(df['labels']))

print(set(df['path'].apply(lambda x: os.path.exists(x))))

df['path'].apply(lambda x: Image.open(x))
# mask = np.column_stack([df['labels'].str.contains(r"'Pastures', 'Broad-leaved forest'", na=False)])
# df2 = df.loc[mask.any(axis=1)]
# print(df2)

num_classes = len(names.values())

en = time.time()
t = en - st
print('Images ready in: {} minutes'.format(int(t/60)))

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

# data_gen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.3)
data_gen = ImageDataGenerator(validation_split=0.3)

print('Flowing training set...')

# directory = path_to_merge,

training_data = data_gen.flow_from_dataframe(
                dataframe = df,
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

validation_data = data_gen.flow_from_dataframe(
                dataframe = df,
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
                    validation_data=validation_data,
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

if __name__ == '__main__':
    rd