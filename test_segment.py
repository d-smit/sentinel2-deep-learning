import numpy as np
import tensorflow as tf
SEED = 12345
tf.set_random_seed(SEED)
np.random.seed(SEED)

import os
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import class_weight

import land_classification as lc

'''
This script tests our segmentation method. After reading in the segmented dataset,
complete with the segment-rank variable, we split into our train/validation/test splits.
'''

root_path = os.getcwd()

# Set with or without the segment_id variable to compare
Segment = True

path_to_model = root_path + '/models/'

print('reading points df')

df = pd.read_csv('seg_points_400k_gsi.csv')
df = df.drop(df.columns[0], axis='columns')
df = df.dropna()

bands = ['B02_1', 'B03_1', 'B04_1', 'B08_1']
indices = ['ndvi']
seg = ['segment_id']

df = lc.calc_indices(df)

def one_hot(df):
    onehot = pd.get_dummies(df['labels_1'])
    df[onehot.columns] = onehot
    return df

def prep_df(df):

    X = df[bands + indices]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)
    X = pd.DataFrame(scaled_data)

    if Segment:
        X['segment_id'] = df['segment_id']

    label_cols = list(df.labels_1.unique())
    y = df[label_cols]

    return X, y

df = one_hot(df)

X, y = prep_df(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def build_model():
    model = Sequential()
    model.add(Dense(400, input_shape=(len(X.columns),), activation="relu"))
    model.add(Dropout(rate=0.2))
    model.add(Dense(300, activation="relu"))
    model.add(Dropout(rate=0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(len(y.columns), activation='softmax'))

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['acc'])

    return model

model = build_model()

class_weights = class_weight.compute_class_weight('balanced',
                                                  df.labels_1.unique(),
                                                  df.labels_1.values)

history = model.fit(X_train,
          y_train,
          epochs=75,
          batch_size=512,
          validation_split=0.20,
          verbose=1,
          class_weight=class_weights)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

'''Then we can make predictions over our testing subset. We predict the
20% held back earlier. '''

y_pred = model.predict(X_test)
y_test = y_test.values

score, acc = model.evaluate(X_test, y_test)
print('Score: {}'.format(score))
print('Accuracy: {}'.format(acc))
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1),
                            target_names=[str(i) for i in y.columns]))

'''
Here we do a simple Random Forest baseline comparison.
'''

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)
forest_preds = rf.predict(X_test)

acc = accuracy_score(y_test, forest_preds)

'''
Now we use the 400k sampled points in order to compare
to the GSI metrics.
'''

gsi_data = pd.read_csv('seg_points_400k_gsi.csv')
gsi_data = gsi_data.dropna()
gsi_data = lc.calc_indices(gsi_data)
gsi_data = one_hot(gsi_data)

labs = np.unique(gsi_data.labels_1)

gsi_pixels = gsi_data[['B02_1', 'B03_1', 'B04_1', 'B08_1', 'ndvi', 'segment_id']].values

scaler = StandardScaler()
scaled_data = scaler.fit_transform(gsi_pixels)
gsi_pixels = pd.DataFrame(scaled_data)

gsi_actuals = gsi_data[labs].values
gsi_preds = model.predict(gsi_pixels)

score, acc = model.evaluate(gsi_pixels, gsi_actuals)
print('Score: {}'.format(score))
print('Accuracy: {}'.format(acc))
print(classification_report(gsi_actuals.argmax(axis=1), gsi_preds.argmax(axis=1),
                            target_names=[str(i) for i in gsi_data[labs].columns]))