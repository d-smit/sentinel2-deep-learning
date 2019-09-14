import numpy as np
import tensorflow as tf
SEED = 12345
tf.set_random_seed(SEED)
np.random.seed(SEED)

import os
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras import optimizers
from keras_preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import collections
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight

''' This script reads in our dataframe of pixel band values and labels, seperately
loading in our array of patches. We then combine these, in order to remove outliers and low
class representations.'''

Server = False

root_path = os.getcwd()
path_to_model = root_path + '/models/'

print('reading points df')

df = pd.read_csv('points_500k_3x3.csv')

print('length of df: {}'.format(len(df)))

values = np.load('patch_arrays_500k_3x3.npz')

patches = values['arr_0']
patches = patches.astype(float)
patches = np.moveaxis(patches, 1, -1)

patch_size = patches[1].shape[0]
patch_count = len(df)
patches = patches.tolist()

df['patches'] = patches

print('Loaded {} {}x{} patches into df'.format((patch_count), patch_size, patch_size))

counter = collections.Counter(df.labels_1.values)
print('class dist before cleaning: {}'.format(counter))

'''
We remove very poorly represented classes
and one-hot encode our labels here.
'''

def clean_and_cut(df):

    df = df.groupby('labels_1').filter(lambda x: x.shape[0] > 100)
    onehot = pd.get_dummies(df['labels_1'])
    df[onehot.columns] = onehot
    class_cols = list(df['labels_1'].unique())
    y = df[class_cols]
    X = np.asarray(df['patches'].values)
    print('One hot encoded labels and removed minor classes (check)')
    print('Cut excess columns, new df shape: {}'.format(df.shape))

    return X, y, df

X, y, clean_df = clean_and_cut(df)

counter = collections.Counter(clean_df.labels_1.values)
print('class dist after cleaning: {}'.format(counter))

X = np.stack(X, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

'''
Here we construct our CNN model, which is fed
using a generator that flows in the patches in batches.
'''

def build_model(in_shape=(X[1].shape), out_shape=len(y.columns)):
     model = Sequential()
     model.add(Conv2D(8, (3, 3), activation='relu', kernel_initializer='uniform', padding='same', input_shape=in_shape))
     model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='uniform', padding='same'))

     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
     model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='uniform', padding='same'))

     model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
     model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='uniform', padding='same'))

     model.add(Flatten())
     model.add(Dense(1000, activation='relu', kernel_initializer='uniform'))

     model.add(Dense(out_shape, activation='softmax'))
     opt = optimizers.Adam(lr=0.0001)

     model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
     return model

model = build_model()

train_gen = ImageDataGenerator()
test_gen = ImageDataGenerator()

training = train_gen.flow(X_train, y_train, batch_size=512)
testing = test_gen.flow(X_test, y_test, batch_size=512)

class_weights = class_weight.compute_class_weight('balanced',
                                                  df.labels_1.unique(),
                                                  df.labels_1.values)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              cooldown=1,
                              patience=5,
                              min_lr=0.000001)

earlystopper = EarlyStopping(monitor='val_acc',
                              patience=10,
                              mode='max')

checkpointer = ModelCheckpoint(path_to_model + '{}x{}_Patch'.format(patch_size, patch_size) +
                               "_{}k".format(patch_count)+
                                "_ms_" +
                                "{epoch:02d}-{val_acc:.3f}" +
                                ".hdf5",
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=False,
                                mode='max')

history = model.fit(X_train,
                    y_train,
                    batch_size=512,
                    epochs=75,
                    validation_split=0.2,
                    callbacks=[earlystopper, checkpointer],
                    class_weight=class_weights)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy for {}x{} Patch'.format(patch_size, patch_size))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

if Server:
    plt.savefig('/home/strathclyde/seg/plots/{}_acc_{}x{}_final_4.jpg'.format(patch_count, patch_size, patch_size))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss for {}x{} Patch'.format(patch_size, patch_size))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

if Server:
    plt.savefig('/home/strathclyde/seg/plots/{}_loss_{}x{}_final_4.jpg'.format(patch_count, patch_size, patch_size))

'''
Here we can evaluate our trained model on our testing subset.
'''

y_pred = model.predict(X_test)
y_test = y_test.values

score, acc = model.evaluate(X_test, y_test)
print('Score: {}'.format(score))
print('Accuracy: {}'.format(acc))

matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

df_cm = pd.DataFrame(matrix, index = [i for i in y.columns],
                  columns = [i for i in y.columns])

plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()
plt.savefig(root_path + '/outputs/confusion_matrix_{}x{}.jpg'.format(patch_size, patch_size))

targs = y.columns
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1),
                            target_names=[str(i) for i in y.columns]))



