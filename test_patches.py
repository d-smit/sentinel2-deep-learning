import os
import numpy as np
import pandas as pd
import rasterio as rio
import pylab as pl
from skimage.util import pad
from skimage.io import imread
from keras.models import load_model
from tqdm import tqdm
import land_classification as lc
from sklearn.metrics import classification_report
import keras.utils

'''
This script acts as our sliding window mechanism for predicting over our AOI.
By loading in our model and AOI, we then iterate over the AOI, taking a patch sized
portion of the AOI at each iteration. The model predicts each image patch and stores
the predictions in an array of the same shape as the AOI. Finally, we can then write this
prediction array as a raster, allowing us to plot the predictions and visually compare
with the ground-truth and a true colour image of the AOI. '''

root_path = os.getcwd()

Server = False
Patch = True
BigEarth = False

''' 
Establishing different paths based on where we were running our code. For experimentation and 
development, we used our own machines, whereas for training final models, the server was used.
'''

if Server:
    path_to_image = root_path + '/masked.tif'
    path_to_gtruth = root_path + '/data/Corine_S2_Proj_2.tif'
    path_to_model = root_path + '/models/7x7_Patch_10k_ms_09-0.666.hdf5'

else:
    path_to_image = root_path + '/data/masked.tif'
    path_to_gtruth = root_path + '/data/Corine_S2_Proj_2.tif'

if Patch:
    path_to_model = root_path + '/models/7x7_Patch_499786k_ms_10-0.664.hdf5'

if BigEarth:
    path_to_model = root_path + '/models/bigearth_ms_05-0.308.hdf5'

'''
Here, we load in our CORINE ground truth, as we want to access the raster 
profile for plotting purposes.'''

scene = rio.open(path_to_gtruth)
profile = scene.profile
data = scene.read(list(pl.arange(scene.count) + 1))

model = load_model(path_to_model)

_, patch_rows, patch_cols, bands = model.layers[0].input_shape

print('Model input shape: {}'.format(model.layers[0].input_shape))

patch_rows_rnd = (int((patch_rows)/2))
patch_cols_rnd = (int((patch_cols)/2))

_, output_classes = model.layers[-1].output_shape

'Now we read in our AOI and pad according to the patch size dimensions.'

img = np.array(imread(path_to_image), dtype=float)

_, unpadded_cols, _ = img.shape

img = pad(img, ((patch_cols, patch_cols),
                (patch_rows, patch_rows),
                    (0, 0)), 'symmetric')

img_rows, img_cols, _ = img.shape

print('Image shape: {}'.format(img.shape))

'''Here we make our empty arrays onto which we will project our predictions. Image_probs
is the same shape of the AOI, which we will fill up by predicting a row's worth of patches,
stored in row_of_patches.
'''

image_probs = np.zeros((img_rows, img_cols, output_classes))

row_of_patches = np.zeros((unpadded_cols, patch_rows, patch_cols, bands))

'''This is our sliding patch mechanism. We iterate over our padded image by row. 
On each row, we access the first pixel by its column position, and crop the image 
around this pixel, forming our patch. Using the index of the pixel position along the row,
we can insert the patch into row_of_patches. This is repeated for every
pixel in the row. After collecting the patches of an entire row, we then make our predictions
for the centre pixels of the patches. This is then repeated for every image row.'''

for row in tqdm(range(patch_rows, img_rows - patch_rows), desc="Processing..."):

    for idx, col in enumerate(range(patch_cols, img_cols-patch_cols)):

        row_of_patches[idx, ...] = img[row-patch_rows_rnd:(row)+patch_rows_rnd,
                                      col-patch_cols_rnd:(col)+patch_cols_rnd,
                                      :]

    # by sliding the patch window along the row, we classify one row per iteration, with a batch size of 1 patch
    classified_row = model.predict(row_of_patches, batch_size=1, verbose=1)

    # adding that classified row to the first row of image_probs
    image_probs[row, patch_cols:img_cols-patch_cols, :] = classified_row

# cut out padding to crop to image boundaries
image_probs = image_probs[patch_rows:img_rows-patch_rows,
                          patch_cols:img_cols-patch_cols, :]

'''
Image_probs now has a prediction in the form of num_class length vector of
probabilities for every pixel in the scene. 
To get the label, we take the position of the highest probability.
'''

try:
    np.savez_compressed('raw_preds_bigearth.npz', image_probs)
except:
    print('cannot compress prediction array')

image_labels = np.argmax(image_probs, axis=2)
image_labels = np.expand_dims(image_labels, axis=0)

'''
To get the probability itself, we sort the vector and take the highest value.
We then need to expand the label and probability arrays back to 3D for plotting.
'''

image_probs = np.sort(image_probs, axis=-1)[..., -1]
image_probs = np.expand_dims(image_probs, axis=0)

print('probs for image: {}'.format(image_probs))

np.savez_compressed('final_probs_7x7.npz', image_probs)
np.savez_compressed('final_labels_7x7.npz', image_labels)

probs = np.load('final_probs_7x7.npz')
probs = probs['arr_0']

labels = np.load('final_labels_7x7.npz')
labels = labels['arr_0']

'''
Need to map prediction labels to label column heads
from our dataset.
'''

actual_labs = np.load('label_heads.npy')
bigearth_labs = np.load('bigearthlabels.npy')

if not BigEarth:
    for row in labels[0]:
        for i, label in enumerate(row):
            row[i] = actual_labs[label]
else:
    for row in labels[0]:
        for i, label in enumerate(row):
            row[i] = bigearth_labs[label]

labels = labels.astype(float)

'''
After transforming our labels to the correct format, we can now write them 
as a raster, using the profile of our CORINE ground-truth, read in earlier.'''

lc.write_raster("outputs/CNN_bigearth_probs.tif", probs, profile)
lc.write_raster("outputs/CNN_bigearth_labels.tif", labels, profile)

# GSI predict

'''
Here we load our trained model and predict
over the 400,000 prediction set.
'''

model = load_model(path_to_model)

data = np.load('patch_arrays_400k_7x7.npz')
gsi_test = data['arr_0']
gsi_test = np.moveaxis(gsi_test, 1, -1)

patch_df = pd.read_csv('points_400k_7x7.csv')

onehot = pd.get_dummies(patch_df['labels_1'])
patch_df[onehot.columns] = onehot

labs = list(np.unique(patch_df.labels_1))
gsi_true = patch_df[labs]

score, acc = model.evaluate(gsi_test, gsi_true)
print('Score: {}'.format(score))
print('Accuracy: {}'.format(acc))

gsi_preds = model.predict(gsi_test)

print(classification_report(gsi_true.values.argmax(axis=1), gsi_preds.argmax(axis=1),
                            target_names=[str(i) for i in patch_df[labs].columns]))


