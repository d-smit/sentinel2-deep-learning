from keras.applications.densenet import DenseNet201 as DenseNet
from keras.applications.vgg16 import VGG16 as VGG
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

import os

dn = False
batch_size = 64

path_split_datasets = '~/Uni/Thesis/EuroSat/Data/split'

path_to_home = os.path.expanduser("~")
path_split_datasets = path_split_datasets.replace("~", path_to_home)

path_to_train = os.path.join(path_split_datasets, 'train')
path_to_valid = os.path.join(path_split_datasets, 'validation')

# need to get sub-directories again

sub_dirs = [sub_dir for sub_dir in os.listdir(path_to_train)]

num_class = len(sub_dirs)

if dn:
    base_model = DenseNet(include_top=False,
                     weights=None,
                     input_shape=(64, 64, 3))
else:
    base_model = VGG(include_top=False,
                     weights=None,
                     input_shape=(64, 64, 3))

top_model = base_model.output
top_model = Flatten()(top_model)

if not DenseNet:

    top_model = Dense(2048, activation='relu')(top_model)
    top_model = Dense(2048, activation='relu')(top_model)

# for final layer of predictions want to use softmax activation on num_classes

''' Predictions in the form of a vector the length 
    of num_classes, representing the probability 
    score for all classes. '''

predictions = Dense(num_class, activation='softmax')(top_model)

''' Inputs formed from convolution layers (base_model),
      and outputting the prediction vectors.'''
model = Model(inputs=base_model.input, outputs=predictions)

# defining image generators

train_datagen = ImageDataGenerator(fill_mode="reflect",
                                   rotation_range=45,
                                   horizontal_flip=True,
                                   vertical_flip=True)

train_generator = train_datagen.flow_from_directory(path_to_train,
                                                    target_size=(64, 64),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
print(train_generator.class_indices)

test_datagen = ImageDataGenerator()

validation_generator = test_datagen.flow_from_directory(path_to_valid,
                                                        target_size=(64, 64),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

model.compile(optimizer='adadelta', loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# generate callback to save best model w.r.t val_categorical_accuracy

if dn:
    file_name = "dense"
else:
    file_name = "vgg"

checkpointer = ModelCheckpoint("/home/david/Uni/Thesis/EuroSat/Models/" + file_name +
                               "_rgb_from_scratch." +
                               "{epoch:02d}-{val_categorical_accuracy:.3f}" +
                               ".hdf5",
                               monitor='val_categorical_accuracy',
                               verbose=1,
                               save_best_only=True,
                               mode='max')

earlystopper = EarlyStopping(monitor='val_categorical_accuracy',
                             patience=50,
                             mode='max')

model.fit_generator(train_generator,
                    steps_per_epoch=3,
                    epochs=1,
                    callbacks=[checkpointer, earlystopper],
                    validation_data=validation_generator,
                    validation_steps=1)
