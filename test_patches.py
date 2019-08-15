import os
import numpy as np
import rasterio as rio
import pylab as pl
from skimage.util import pad
from skimage.io import imread
from keras.models import load_model
from tqdm import tqdm
import land_classification as lc

root_path = os.getcwd()

Server = True
Server = False

if Server:
    path_to_image = root_path + '/data/masked.tif'
    path_to_model = root_path + '/models/CNNPatch_rgb_01-0.449.hdf5'

else:
    path_to_image = root_path + '/data/masked.tif'
    path_to_model = root_path + '/models/CNNPatch_rgb_01-0.449.hdf5'

scene = rio.open(path_to_image)
profile = scene.profile
data = scene.read(list(pl.arange(scene.count) + 1))

model = load_model(path_to_model)

_, patch_rows, patch_cols, bands = model.layers[0].input_shape

patch_rows_rnd = (int((patch_rows)/2))
patch_cols_rnd = (int((patch_cols)/2))

_, output_classes = model.layers[-1].output_shape

img = np.array(imread(path_to_image), dtype=float)

_, unpadded_cols, _ = img.shape

img = pad(img, ((patch_cols, patch_cols),
                (patch_rows, patch_rows),
                    (0, 0)), 'symmetric')

num_rows, num_cols, _ = img.shape

image_probs = np.zeros((num_rows, num_cols, output_classes))

row_of_patches = np.ones((unpadded_cols, patch_rows, patch_cols, bands))

for row in tqdm(range(patch_rows, num_rows-patch_rows), desc="Processing..."):

    for idx, col in enumerate(range(patch_cols, num_cols-patch_cols)):

        row_of_patches[...] = img[row-patch_rows_rnd:(row+1)+patch_rows_rnd,
                                  col-patch_cols_rnd:(col+1)+patch_cols_rnd,
                                  :]

    classified_row = model.predict(row_of_patches, batch_size=1, verbose=1)
    image_probs[row, patch_cols:num_cols-patch_cols, :] = classified_row

print('probs for image: {}'.format(image_probs))
np.savez_compressed('scene_probs_ex.npz', image_probs)

probs = np.load('scene_probs.npz')
probs = probs['arr_0']

probs = np.moveaxis(probs, 2, 0)
probp = probs.max(axis=0)
probp = np.expand_dims(probp, axis=0)

lc.write_raster("outputs/lc_10m_5x5_probs2.tif", probp, profile)


