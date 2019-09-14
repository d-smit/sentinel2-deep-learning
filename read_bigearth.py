import collections
from glob import glob
import pandas as pd
import os
import numpy as np
from numpy import asarray
from numpy import savez_compressed
import time
import random
import json
import scipy.misc
import itertools
from tqdm import tqdm
from itertools import chain
from PIL import Image
# import land_classification as lc
import rasterio as rio
import pylab as pl
st = time.time()

'''
This script reads in the BigEarth dataset, merges the 4 bands we were considering
and grabs the GSI relevant labels. We then compress this and store it as an array.'''

root_path = os.getcwd()

Server = False
# Server = True

if Server:
    path_to_images = root_path + '/DATA/bigearth2/BigEarthNet-v1.0/'
    # path_to_images = root_path + '/DATA/bigearth/sample/'
    path_to_merge = root_path + '/DATA/bigearth/check2/'
    dst = '/DATA'
    with open('/home/strathclyde/DATA/corine_labels.json') as jf:
        names = json.load(jf)

else:
    path_to_images = root_path + '/data/sample/'
    path_to_merge = root_path + '/data/merge/'
    dst = '/data'
    with open('data/corine_labels.json') as jf:
        names = json.load(jf)

'''Here we designate the gsi classes which are present in the AOI, we are not 
concerned with the remaining patches.'''

gsi_labels = [12, 18, 2, 23, 11, 1, 10, 3, 25, 21, 8, 6, 4, 29, 9, 41, 0]
gsi_classes = [v for k,v in names.items() for l in gsi_labels if names[str(l)] == names[k]]

def clean_patches(dst):
    patches = [patches for patches in os.listdir(path_to_images)]
    patches = patches[:int(len(patches)/(3/2))]
    valid_split = 0.3
    print('Unprocessed patch count: {}'.format(len(patches)))

    cloud_patches = pd.read_csv(root_path + '{}/patches_with_cloud_and_shadow.csv'.format(dst), 'r', header=None)
    snow_patches = pd.read_csv(root_path + '{}/patches_with_seasonal_snow.csv'.format(dst), 'r', header=None)

    patch_set = set(cloud_patches[0]).union(set(snow_patches[0]))
    patches = set(patches)
    patches = list(patches - patch_set)
    print('{} clean patches...'.format(len(patches)))
    split_point = int(len(patches) * valid_split)

    return patches, split_point

def merge_label_patches(patches, bands = ['B02', 'B03', 'B04', 'B08'], nodata=-9999):

    random.shuffle(patches)
    image_paths = []
    image_labels = []

    print('Merging and tagging...')

    for i in tqdm(range(0, len(patches) - 1)):

        if Server:
            with open('/home/strathclyde/DATA/bigearth2/BigEarthNet-v1.0/{}/{}_labels_metadata.json' \
                      .format(patches[i], patches[i])) as js:
                meta = json.load(js)
        else:
            with open('data/sample/{}/{}_labels_metadata.json' \
                      .format(patches[i], patches[i])) as js:
                meta = json.load(js)

        # remove non-GSI classes

        labels = meta.get('labels')
        labels = [i for i in labels if i in gsi_classes]

        if labels == []:
            continue
        else:
            pass

        labels = labels[0]

        '''
        We need to merge the bands together. This is done by opening each as a
        raster, stacking the arrays and saving to the new 3-D array to a list.'''

        tifs = glob(path_to_images + '/{}/*.tif'.format(patches[i]), recursive=True)
        band_tifs = [tif for tif in tifs for band in bands if band in tif]

        bands_to_stack = list(map(rio.open, band_tifs))
        img = pl.stack(list(map(lambda x: x.read(1).astype(pl.int16), bands_to_stack)))
        img = np.moveaxis(img, 0, -1)

        image_paths.append(img)
        image_labels.append(labels)

    d = {'labels': image_labels}
    df = pd.DataFrame(d)

    return df, image_paths

'''
Here we map the CORINE Land-Types to integer values
'''

def map_labels(df):
    names2 = {v: int(k) for k, v in names.items()}
    class_to_label = []
    for entry in df.labels.values:
        entry = names2[entry]
        class_to_label.append((entry))

    df['labels'] = pd.Series(data=class_to_label)
    lst = df['labels'].values
    return df

def one_hot(df):
    onehot = pd.get_dummies(df['labels'])
    df[onehot.columns] = onehot
    return df

def load_dataset():

    cleanpatch, split_point = clean_patches(dst)
    df, images = merge_label_patches(cleanpatch)
    df = map_labels(df)
    df = one_hot(df)

    counter=collections.Counter(df.labels.values)
    print('class dist: {}'.format(counter))

    X = asarray(images, dtype='uint8')
    y = df.iloc[:,1:].values

    return X, y, df

st=time.time()
X, y, df = load_dataset()
savez_compressed('bigearth.npz', X, y)
en=time.time()
print('dataset ready in {}'.format(en-st))

