from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import random as rn
rn.seed(1)

import os
import time
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

from numpy import load

Server = False
Server = True

'''
This script reads in BigEarth and feeds it into our CNN. We build our CNN and flow the our images
into it using a generator. We also plot the training history and create a classification report. '''

print('Loading dataset...')

st = time.time()

data = load('bigearth.npz')

X, y = data['arr_0'], data['arr_1']

print('Loaded: ', X.shape, y.shape)
print('Imports done')

root_path = os.getcwd()

path_to_model = root_path + '/bigearth/model2/'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

en = time.time()
t = en - st
print('Images ready in: {} minutes'.format(int(t/60)))

def build_model(in_shape=(120, 120, 4), out_shape=len(y[1])):

 	model = Sequential()
 	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=in_shape))
 	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 	model.add(MaxPooling2D((2, 2)))
 	model.add(Dropout(0.2))
 	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 	model.add(MaxPooling2D((2, 2)))
 	model.add(Dropout(0.2))
 	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 	model.add(MaxPooling2D((2, 2)))
 	model.add(Flatten())
 	model.add(Dense(200, activation='relu', kernel_initializer='he_uniform'))
 	model.add(Dropout(0.2))
 	model.add(Dense(out_shape, activation='softmax'))

 	opt = Adam(lr=0.0005)
 	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

 	return model

model = build_model()

'''
Using a generator to feed in batches of BigEarth and rescaling.
'''

train_gen = ImageDataGenerator(rescale=1.0/255.0)
test_gen = ImageDataGenerator(rescale=1.0/255.0)

training = train_gen.flow(X_train, y_train, batch_size=512)
testing = test_gen.flow(X_test, y_test, batch_size=512)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              cooldown=1,
                              patience=5,
                              min_lr=0.001)

earlystopper = EarlyStopping(monitor='val_categorical_accuracy',
                              patience=10,
                              mode='max')

checkpointer = ModelCheckpoint(path_to_model + 'bigearth' +
                                "_ms_" +
                                "{epoch:02d}-{val_categorical_accuracy:.3f}" +
                                ".hdf5",
                                monitor='val_categorical_accuracy',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=False,
                                mode='max')

history = model.fit_generator(training,
                            steps_per_epoch=len(training),
                            epochs=25,
                            validation_data=testing,
                            callbacks=[earlystopper, checkpointer],
                            validation_steps=len(testing))

history.history.keys()
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('BigEarthCNN Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

if Server:
    plt.savefig('/home/strathclyde/DATA/plots/big_earth_acc_dropout_barebones_act.jpg')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('BigEarthCNN Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

if Server:
    plt.savefig('/home/strathclyde/DATA/plots/bigearth_loss_dropout_barebones_act.jpg')

y_pred = model.predict(X_test)
y_test = y_test

score, acc = model.evaluate(X_test, y_test)
print('Score: {}'.format(score))
print('Accuracy: {}'.format(acc))

gsi_labels = [12, 18, 2, 23, 11, 1, 10, 3, 25, 21, 8, 6, 4, 29, 9, 41, 0]

print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1),
                            target_names=[str(i) for i in gsi_labels]))



