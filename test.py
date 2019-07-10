import numpy as np
from skimage.io import imread
from keras.models import load_model
import time
# import gdal

# inputs
path_to_test_image = '/home/david/Uni/Thesis/EuroSat/Data/test/tci_medium.tif'
path_to_model = '/home/david/Uni/Thesis/EuroSat/Models/vgg_net/vgg_rgb_transfer_final.206-0.977.hdf5'

# result paths
path_to_label_image = '/home/david/Uni/Thesis/EuroSat/Data/output/vgg_label.tif'
path_to_prob_image =  '/home/david/Uni/Thesis/EuroSat/Data/output/vgg_prob.tif'

# get image for input
image = np.array(imread(path_to_test_image))

# image dims
num_rows, num_cols, num_channels = image.shape

# get model
model = load_model(path_to_model)

# input dimensions, these will be the patch dimensions
_, input_height, input_width, input_channels = model.layers[0].input_shape

# output dimensions, we've got 10 classes to predict
_, output_classes = model.layers[-1].output_shape

# output dimensions to project probabilities onto, same shape as image
class_probs = np.zeros((num_rows, num_cols, output_classes))

# output array to store all of our scene patches
patches = np.zeros((num_cols*num_rows, input_height, input_width, input_channels))

# now need to convolve over the image grabbing the patches
for row in range(input_height, num_rows - input_height):

    for idx, col in enumerate(range(input_width, num_cols - input_width)):

        patches[idx, ...] = image[row-int(input_height/2):row+int(input_height/2),
                                  col-int(input_height/2):col+int(input_height/2), :]


print('Making patch-wise predictions...')
start = time.time()

# get an array of len(rows*pixels) arrays each of len(num_classes) per patch showing preds
patches_clfd = model.predict(patches)

end = time.time()
time_taken=end-start
print('Time taken: {} minutes'.format(time_taken/60))

# reshaping patch prediction array to image height and width and depth=num_classes
patches_clfd = patches_clfd.reshape(image.shape[0],
                                    image.shape[1],
                                    output_classes)

# filling empty array with patch predictions
class_probs[...] = patches_clfd

# getting position of top prediction for each patch
class_labels = np.argmax(class_probs, axis=-1)

# sorting to get actual top prediction for each patch
class_probs = np.sort(class_probs, axis=-1)[..., -1]


