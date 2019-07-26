from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import random as rn
rn.seed(1)

from glob import glob
import pandas as pd
import rasterio as rio
import pylab as pl
import os
import numpy as np
import time
import random
import json
import scipy.misc

st = time.time()
root_path = os.getcwd()

Server = True

if Server:
    path_to_images = root_path + '/DATA/bigearth/sample/'
    path_to_merge = root_path + '/DATA/bigearth/merge/'
    
    with open('/home/strathclyde/DATA/corine_labels.json') as jf:
        names = json.load(jf)

else:
    path_to_images = root_path + '/data/sample/'
    path_to_merge = root_path + '/data/merge/'
    
    with open('data/corine_labels.json') as jf:
        names = json.load(jf)

patches = [patches for patches in os.listdir(path_to_images)]

# def build_dirs():
#     for value in names.values():
#         if not os.path.isdir(path_to_train + value):
#             os.makedirs(path_to_train + value)
#         if not os.path.isdir(path_to_validation + value):
#             os.makedirs(path_to_validation + value)
#     for split in os.listdir(path_to_split):
#         split = split + '/'
#         for sub_dir in os.listdir(os.path.join(path_to_split, split)):
#             sub_dir = sub_dir + '/'
#             split_path = path_to_split + split
#             if not os.listdir(os.path.join(split_path, sub_dir)):
#                 continue
#             else:
#                 for file in os.listdir(os.path.join(split_path, sub_dir)):
#                     filepath = split_path + sub_dir + file
#                     os.remove(filepath)

def get_patches(patches):

    valid_split = 0.3
    print('Raw patch count: {}'.format(len(patches)))

    if not Server:
        dst = '/data'
    else:
        dst = '/DATA'

    cloud_patches = pd.read_csv(root_path + '{}/patches_with_cloud_and_shadow.csv'.format(dst), 'r', header=None)
    snow_patches = pd.read_csv(root_path + '{}/patches_with_seasonal_snow.csv'.format(dst), 'r', header=None)
    bad_patches = pd.concat([cloud_patches, snow_patches], axis=1)

    print('Purging cloud, shadow and snow patches...')

    patches = [patch for patch in patches if not patch in bad_patches.values]

    # for patch in patches:
    #     bad_patches.apply(lambda x: patches.remove(x==patch))
    print('Processing {} clean patches...'.format(len(patches)))
    
    split_point = int(len(patches) * valid_split)

    return patches, split_point

# patches, split_point = get_patches(patches)

def read_patch(split_point, bands = ['B02', 'B03', 'B04'], nodata=-9999):

    ''' Returns a NumPy array for each patch,
    consisting of the four bands'''

    print('Shuffling patches...')
    random.shuffle(patches)

    cols = ['path', 'labels']
    index = range(0, len(patches) - 1)
    df = pd.DataFrame(index=index, columns=cols)

    print('Merging RGB bands...')
    class_rep = []

    for i in range(0, len(patches) - 1):

        tifs = glob(path_to_images + '{}/*.tif'.format(patches[i]), recursive=True)
        band_tifs = [tif for tif in tifs for band in bands if band in tif]

        if Server:
            with open('/home/strathclyde/DATA/bigearth/sample/{}/{}_labels_metadata.json'\
                      .format(patches[i], patches[i])) as js:
                meta = json.load(js)
        else:
            with open('data/sample/{}/{}_labels_metadata.json' \
                      .format(patches[i], patches[i])) as js:
                meta = json.load(js)

        labels = meta.get('labels')

        label = labels[0] + '/'
        if label == 'Transitional woodland/shrub/':
            label = 'Transitional woodland or shrub/'
        class_rep.append(label)

        band_tifs.sort()

        files2rio = list(map(rio.open, band_tifs))
        data = pl.stack(list(map(lambda x: x.read(1).astype(pl.int16), files2rio)))
        data = np.moveaxis(data, 0, 2)

       # scipy.misc.toimage(data[...]).save(path + label + patches[i] + '.jpeg')

        scipy.misc.toimage(data[...]).save(path_to_merge + patches[i] + '.jpg')

        # if i==6:
        #     print(path_to_merge + patches[i])

        # df.iloc[i] = [patches[i] + '.jpg', ', '.join(labels)]
        df.iloc[i] = [path_to_merge + patches[i] + '.jpg', str(labels).strip('[]')]

    # print('Patches sorted into classes...')

    class_count = len(set(class_rep))

    print('Ready to train on {} RGB patches belonging to {}/{} classes.' \
      .format(len(patches), class_count, len(names)))


    return class_count, df

# class_count, df = read_patch(split_point)




