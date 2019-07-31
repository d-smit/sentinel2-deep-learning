import os
from skimage.io import imread
import numpy as np
from skimage.util import pad

root_path = os.getcwd()

path_to_image = root_path + '/data/test/tci_medium.tif'
path_to_model = '/models/model_name.hdf5'

patch_wid = 120
patch_hgt = 120

img = np.array(imread(path_to_image), dtype=float)

num_rows, num_cols, num_channels = img.shape

# need to pad image based on input to model i.e. 120 by 120 patch sizes

img = pad(img, ((patch_wid, patch_wid),
                (patch_hgt, patch_hgt),
                (0, 0)), 'symmetric')

