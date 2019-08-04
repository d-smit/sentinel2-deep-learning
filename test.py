import os
from skimage.io import imread
import numpy as np
from skimage.util import pad
from keras.models import load_model
from tqdm import tqdm

root_path = os.getcwd()
Server = True
Server = False

if Server:
    path_to_image = root_path + '/DATA/bigearth/test/tci_medium.tif'
    path_to_model = root_path + '/models/bigearth_rgb_02-0.488.hdf5'

else:
    path_to_image = root_path + '/data/test/tci_medium.tif'
    path_to_model = root_path + '/data/models/bigearth_rgb_02-0.488.hdf5'

model = load_model(path_to_model)
_, input_rows, input_cols, channels = model.layers[0].input_shape
_, output_classes = model.layers[-1].output_shape

img = np.array(imread(path_to_image), dtype=float)

_, unpadded_cols, _ = img.shape

img = pad(img, ((input_cols, input_cols),
                (input_rows, input_rows),
                (0, 0)), 'symmetric')

num_rows, num_cols, _ = img.shape

image_probs = np.zeros((num_rows, num_cols, output_classes))

# want to predict row by row, so going to make 4d array
row_of_patches = np.zeros((unpadded_cols, input_rows, input_cols, channels))

for row in tqdm(range(input_rows, num_rows-input_rows), desc="Processing..."):

    for idx, col in enumerate(range(input_cols, num_cols-input_cols)):

        row_of_patches[...] = img[row-int(input_rows/2):row+int(input_rows/2),
                                  col-int(input_cols/2):col+int(input_cols/2),
                                  :]

    classified_row = model.predict(row_of_patches, batch_size=1, verbose=1)
    image_probs[row, input_cols:num_cols-input_cols, :] = classified_row

