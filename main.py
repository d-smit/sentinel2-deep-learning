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

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation
from keras import optimizers
from keras.optimizers import SGD
from PIL import Image
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

from keras.applications.densenet import DenseNet201 as DenseNet
from keras.models import Model

from keras import backend

# import read_data_df as rd


# load prepared planet dataset
from numpy import load

print('Loading dataset...')
data = load('bigearth.npz')
X, y = data['arr_0'], data['arr_1']
print('Loaded: ', X.shape, y.shape)
print('Imports done')

root_path = os.getcwd()

Server = False
Server = True

cv = False
# cv = True

# if Server:
#     path_to_images = root_path + '/DATA/bigearth/dump/sample/'
#     path_to_model = root_path + '/DATA/bigearth/model/'
#     with open('/home/strathclyde/DATA/corine_labels.json') as jf:
#         names = json.load(jf)

# else:
#     path_to_images = root_path + '/data/sample/'
#     path_to_model = root_path + '/data/models/'

#     with open('data/corine_labels.json') as jf:
#         names = json.load(jf)

st = time.time()

# patches = [patches for patches in os.listdir(path_to_images)]
# patches, split_point = rd.get_patches(patches)
# print('patch count: {}'.format(len(patches)))
# print('split point: {}'.format(split_point))

# split_point=3751
# df, class_count = rd.read_patch()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

def fbeta(y_true, y_pred, beta=2):
	# clip predictions
	y_pred = backend.clip(y_pred, 0, 1)
	# calculate elements
	tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
	fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
	fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
	# calculate precision
	p = tp / (tp + fp + backend.epsilon())
	# calculate recall
	r = tp / (tp + fn + backend.epsilon())
	# calculate fbeta, averaged across each class
	bb = beta ** 2
	fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
	return fbeta_score

def fold_df(df, k):
    X = df.path
    y = df.labels
    folds = KFold(n_splits=k, shuffle=True, random_state=1).split(X, y)
    return folds, X, y

# if cv:
#     k = 7
#     folds, X_train, y_train = fold_df(df, k)

# class_rep = list(chain.from_iterable(df['labels']))
# counter=collections.Counter(class_rep)

# print('class dist: {}'.format(counter))
# print('class count: {}'.format(class_count))

en = time.time()
t = en - st
print('Images ready in: {} minutes'.format(int(t/60)))

def build_model(in_shape=(120, 120, 3), out_shape=6):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=in_shape))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(out_shape, activation='sigmoid'))

	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['categorical_accuracy'])

	return model

model = build_model()
train_gen = ImageDataGenerator(rescale=1.0/255.0, horizontal_flip=True, vertical_flip=True, rotation_range=90)
test_gen = ImageDataGenerator(rescale=1.0/255.0)

training = train_gen.flow(X_train, y_train, batch_size=128)
print('Training set size: {}'.format(len(training)))

testing = test_gen.flow(X_test, y_test, batch_size=128)
print('testing set size: {}'.format(len(testing)))

history = model.fit_generator(training,
                            steps_per_epoch=2000,
                            epochs=25,
                            validation_data=testing,
                            validation_steps=1000)



# evaluate model

loss, fbeta = model.evaluate_generator(testing, steps=len(testing), verbose=1)

print('> loss=%.3f, fbeta=%.3f' % (loss, fbeta))


# reduce_lr = ReduceLROnPlateau(monitor='val_loss',
#                               factor=0.2,
#                               cooldown=1,
#                               patience=2,
#                               min_lr=0.0001)

# earlystopper = EarlyStopping(monitor='val_categorical_accuracy',
#                              patience=5,
#                              mode='max')

# print('Normalizing images...')

# def preprocessing(x):

#     mean = [87.845, 96.965, 103.947]
#     std = [23.657, 16.474, 13.793]

#     for idx, mean_value in enumerate(mean):
#         x[..., idx] -= mean_value
#         x[..., idx] /= std[idx]

#     return x

# # raw_labels = df.labels.values.tolist()
# # raw_labels = [l for sub_l in raw_labels for l in sub_l]

# # class_weightings = class_weight.compute_class_weight('balanced',
# #                                                       list(counter),
# #                                                       raw_labels)

# # print('Class weights {}'.format(class_weightings))

# # # preprocessing_function=preprocessing,
# # # fill_mode="reflect",
# # #                             rotation_range=45,
# # #                             horizontal_flip=True,
# # #                             vertical_flip=True,
# # #                           validation_split=0.2
# # gen = ImageDataGenerator()

# # if cv:
# #     for j, (train_idx, val_idx) in enumerate(folds):

# #         print('\nFold ',j)

# #         X_train_cv = X_train[train_idx]
#         # y_train_cv = y_train[train_idx]

#         # dict1 = {'path': X_train_cv, 'labels': y_train_cv}
#         # df1 = pd.DataFrame(dict1)
#         # # print(df1)

#         # X_valid_cv = X_train[val_idx]
#         # y_valid_cv = y_train[val_idx]

#         # dict2 = {'path': X_valid_cv, 'labels': y_valid_cv}
#         # df2 = pd.DataFrame(dict2)
#         # # generator = gen.flow(X_train_cv, y_train_cv, batch_size = 64)

#         # generator = gen.flow_from_dataframe(
#         #                 dataframe = train,
#         #                 x_col = 'path',
#         #                 y_col = 'labels',
#         #                 seed = 1,
#         #                 target_size = (120,120),
#         #                 subset = 'training',
#         #                 class_mode = 'categorical',
#         #                 batch_size = 64,
#         #                 shuffle = True)

#         # v_generator = gen.flow_from_dataframe(
#         #                 dataframe = train,
#         #                 x_col = 'path',
#         #                 y_col = 'labels',
#         #                 seed = 1,
#         #                 target_size = (120,120),
#         #                 subset = 'validation',
#         #                 class_mode = 'categorical',
#         #                 batch_size = 64,
#         #                 shuffle = True)

#         # history = model.fit_generator(generator,
#         #                     steps_per_epoch=1,
#         #                     epochs=1,
#         #                     validation_data=v_generator,
#         #                     validation_steps=1,
#         #                     class_weight = class_weightings)

#         # # print(model.evaluate(X_valid_cv, y_valid_cv))

#         # # test_arrays = np.empty(len(X_valid_cv))

#         # corines = list(counter.keys())

#         # predictions = []

#         # for i in test.path.values:
#         #     img_ar = Image.open(i)
#         #     img_ar = np.expand_dims(img_ar, axis=0)
#         #     pred = model.predict(img_ar)
#         #     # pred = [l for l in pred]
#         #     # pred = np.argmax(pred, axis=1)
#         #     pred_abs = (pred > 0.5).astype(np.int)
#         #     pred_abs = pred_abs.tolist()
#         #     pred_abs = list(chain.from_iterable(pred_abs))

#         #     for i in (range(0, len(pred_abs))):
#         #         if pred_abs[i]:
#         #             pred_abs[i]=corines[i]
#         #     pred_abs = list(filter((0).__ne__, pred_abs))
#         #     predictions.append(pred_abs)

#         # predictions



# #         preds = [l.tolist() for sl in predictions for l in sl]

# #         actuals = [l for l in test.labels.values]# for l in sl]
# #         # actuals = MultiLabelBinarizer().fit_transform(actuals)

# #         cf = confusion_matrix(test.iloc[:, 2:], np.array(preds))

# #         cf = confusion_matrix(actuals, predictions)

# #         print(cf)
# # else:

# #     print('Flowing training set...')

# #     training_data = gen.flow_from_dataframe(
# #                     dataframe = train,
# #                     x_col = 'path',
# #                     y_col = 'labels',
# #                     subset = "training",
# #                     seed = 1,
# #                     target_size = (120,120),
# #                     class_mode = 'categorical',
# #                     batch_size = 32,
# #                     shuffle = True)
    
# #     train_cls = training_data.class_indices
# #     train_rep = training_data.classes
# #     train_rep = list(chain.from_iterable(train_rep))
# #     counter=collections.Counter(train_rep)
    
# #     print('Training indices: {}'.format(train_cls))
# #     print('training class count: {}'.format(counter))

# #     print('Flowing validation set...')
    
# #     validation_data = gen.flow_from_dataframe(
# #                     dataframe = train,
# #                     x_col = 'path',
# #                     y_col = 'labels',
# #                     subset = "validation",
# #                     seed = 1,
# #                     target_size = (120,120),
# #                     class_mode = 'categorical',
# #                     batch_size = 32,
# #                     shuffle = True)
    
# #     valid_cls = validation_data.class_indices
# #     valid_rep = validation_data.classes
# #     valid_rep = list(chain.from_iterable(valid_rep))
# #     counter = collections.Counter(valid_rep)
    
# #     print('Validation indices: {}'.format(validation_data.class_indices))
# #     print('valid class count: {}'.format(counter))

# #     checkpointer = ModelCheckpoint(path_to_model + 'bigearth' +
# #                                "_rgb_" +
# #                                "{epoch:02d}-{val_categorical_accuracy:.3f}" +
# #                                ".hdf5",
# #                                monitor='val_categorical_accuracy',
# #                                verbose=1,
# #                                save_best_only=True,
# #                                save_weights_only=False,
# #                                mode='max')

# #     history = model.fit_generator(training_data,
# #                                     steps_per_epoch = 2000,
# #                                     epochs = 15,
# #                                     validation_data = validation_data,
# #                                     validation_steps = 1000,
# #                                     callbacks=[reduce_lr, earlystopper, checkpointer],
# #                                     class_weight = class_weightings)

# #     # corines = list(counter.keys())

# #     # predictions = []

# #     # for i in test.path.values:
# #     #     img_ar = Image.open(i)
# #     #     img_ar = np.expand_dims(img_ar, axis=0)
# #     #     pred = model.predict(img_ar)
# #     #     # pred = [l for l in pred]
# #     #     # pred = np.argmax(pred, axis=1)
# #     #     pred_abs = (pred > 0.5).astype(np.int)
# #     #     pred_abs = pred_abs.tolist()
# #     #     pred_abs = list(chain.from_iterable(pred_abs))

# #     #     for i in (range(0, len(pred_abs))):
# #     #         if pred_abs[i]:
# #     #             pred_abs[i]=corines[i]
# #     #     pred_abs = list(filter((0).__ne__, pred_abs))
# #     #     predictions.append(pred_abs)

# #     # predictions
# #     # preds = [l.tolist() for sl in predictions for l in sl]

# #     # actuals = [l for l in test.labels.values]# for l in sl]
# #     # # actuals = MultiLabelBinarizer().fit_transform(actuals)

# #     # cf = confusion_matrix(test.iloc[:, 2:], np.array(preds))

# #     # cf = confusion_matrix(actuals, predictions)

# #     # print(cf)

# # if Server:
# #     plot_spot = root_path + '/DATA/bigearth/output/acc'

# # else:
# #     plot_spot = root_path + '/data/output/acc'

# # plt.plot(history.history['categorical_accuracy'])
# # plt.plot(history.history['val_categorical_accuracy'])
# # plt.title('model accuracy')
# # plt.ylabel('accuracy')
# # plt.xlabel('epoch')
# # plt.legend(['train', 'test'], loc='lower right')
# # plt.savefig(plot_spot)
# # # plt.show(block=True)

# # plt.plot(history.history['loss'])
# # plt.plot(history.history['val_loss'])
# # plt.title('model loss')
# # plt.ylabel('loss')
# # plt.xlabel('epoch')
# # plt.legend(['train', 'test'], loc='upper right')
# # plt.savefig('/home/strathclyde/DATA/bigearth/output/loss_15_10_0.1.jpg')
# # # plt.show(block=True)

# # if __name__ == '__main__':
# #     rd
# # plt.bar(range(len(counter)), list(counter.values()), align='center')
# # plt.xticks(range(len(counter)), list(counter.keys()))
# # plt.title('Training classes')
# # plt.ion()
# # plt.show()
#             # pred = pred.argsort()[-3:][::-1]

# # df['l2'] = df['labels'].apply(lambda x: str(x).strip('[]'))

# # plt.bar(range(len(counter)), list(counter.values()), align='center')
# # plt.xticks(range(len(counter)), list(counter.keys()))
# # plt.title('Training classes')
# # plt.ion()
# # plt.show()

# # srv_dic = {'25': 5667, '12': 4636, '23': 3807, '18': 2627, '41': 2374, '2': 1710, '1': 470, '3': 281, '11': 112, '4': 82, '10': 33, '6': 26}
# # srv_dic= {'25': 5667, '12': 4636, '23': 3807, '18': 2627, '41': 2374, '2': 1710, '1': 470, '3': 281, '11': 112, '4': 82, '10': 33, '6': 26}

# # if Server:
# #     classes = [k for k in srv_dic]
# # else:
# #     classes = ['10', '12', '2', '23', '25', '3', '41']

#     #classes = [i for i in range(1,6) if i != 4]
#     # classes = set(df.labels.values.tolist())# For local sample dataset
