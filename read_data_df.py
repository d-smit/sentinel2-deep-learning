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
import itertools

import timeit

# from sklearn.preprocessing import MultiLabelBinarizer

st = time.time()
root_path = os.getcwd()

Server = False
# Server = True

if Server:
    path_to_images = root_path + '/DATA/bigearth/dump/sample/'
    path_to_merge = root_path + '/DATA/bigearth/merge2/'

    with open('/home/strathclyde/DATA/corine_labels.json') as jf:
        names = json.load(jf)

else:
    path_to_images = root_path + '/data/sample/'
    path_to_merge = root_path + '/data/merge/'
    
    with open('data/corine_labels.json') as jf:
        names = json.load(jf)

patches = [patches for patches in os.listdir(path_to_images)]

def get_patches(patches):

    st = time.time()

    valid_split = 0.3
    print('Unprocessed patch count: {}'.format(len(patches)))

    dst = '/DATA' if Server == True else '/data'
    
    cloud_patches = pd.read_csv(root_path + '{}/patches_with_cloud_and_shadow.csv'.format(dst), 'r', header=None)
    snow_patches = pd.read_csv(root_path + '{}/patches_with_seasonal_snow.csv'.format(dst), 'r', header=None)
    bad_patches = pd.concat([cloud_patches, snow_patches], axis=1)

    print('Purging cloud, shadow and snow patches...')
    patches = [patch for patch in patches if not patch in bad_patches.values]

    print('{} clean patches...'.format(len(patches)))

    split_point = int(len(patches) * valid_split)

    en = time.time()

    print('Purge took: {} secs '.format(float(en-st)))

    return patches, split_point


def read_patch(split_point, bands = ['B02', 'B03', 'B04'], nodata=-9999):

    ''' Returns a NumPy array for each patch,
    consisting of the four bands'''
    st = time.time()
    print('Shuffling patches...')
    random.shuffle(patches)

    cols = ['path', 'labels']
    index = range(0, len(patches) - 1)
    df = pd.DataFrame(index=index, columns=cols, dtype=object)

    print('Merging RGB bands...')

    path_col = []
    label_col = []

    for i in range(0, len(patches) - 1):

        tifs = glob(path_to_images + '{}/*.tif'.format(patches[i]), recursive=True)
        band_tifs = [tif for tif in tifs for band in bands if band in tif]

        if Server:
            with open('/home/strathclyde/DATA/bigearth/dump/sample/{}/{}_labels_metadata.json' \
                      .format(patches[i], patches[i])) as js:
                meta = json.load(js)
        else:
            with open('data/sample/{}/{}_labels_metadata.json' \
                      .format(patches[i], patches[i])) as js:
                meta = json.load(js)

        labels = meta.get('labels')

        band_tifs.sort()
        files2rio = list(map(rio.open, band_tifs))
        data = pl.stack(list(map(lambda x: x.read(1).astype(pl.int16), files2rio)))
        data = np.moveaxis(data, 0, 2)

        scipy.misc.toimage(data[...]).save(path_to_merge + patches[i] + '.jpg')

        path_col.append(path_to_merge + patches[i] + '.jpg')
        label_col.append(labels)

    d = {'path': path_col, 'labels': label_col}
    df = pd.DataFrame(d)


    lst = df['labels'].values
    classes = list(itertools.chain.from_iterable(lst))
    en = time.time()
    print('merged bands in {} sec'.format(float(en-st)))

    return df, len(set(classes))

split_point = 299
df, class_count = read_patch(split_point)
# cldict = {
#         1: [names[x] for x in [str(i) for i in range(1,12)]],                   # Artificial surfaces: 1 - 11
#         2: [names[x] for x in [str(i) for i in range(12,23)]],                  # Agriculture: 12 - 22
#         3: [names[x] for x in [str(i) for i in range(23,26)]],                    # Forest: 23, 24, 25
#         4: [names[x] for x in [str(i) for i in range(26,30)]],                # Vegetation: 26-29
#         5: [names[x] for x in [str(i) for i in range(30,35)]],            # Open space with little veg: 30-34
#         6: [names[x] for x in [str(i) for i in range(35,45)]]               # Water: 35 - 44
#     }

ldict = {
        1: [i for i in range(1,12)],                   # Artificial surfaces: 1 - 11
        2: [i for i in range(12,23)],                  # Agriculture: 12 - 22
        3: [i for i in range(23,26)],                    # Forest: 23, 24, 25
        4: [i for i in range(26,30)],                # Vegetation: 26-29
        5: [i for i in range(30,35)],            # Open space with little veg: 30-34
        6: [i for i in range(35,45)]               # Water: 35 - 44
    }

# # Need to convert df.labels to names.key instead of names.value


names2 = {v: k for k, v in names.items()}


# df["labels"] = df["labels"].apply(lambda x: str(x).strip('[]'))
# df['labels'] = df['labels'].map(pd.Series(names2))
# df.labels
# df['labels'].replace(names2)
# df = df.replace(df.labels, names)

# df.replace({'labels': names2})

# df['labels'].put(names2.keys(), names2.values())
med_col = pd.Series(data=np.empty(df.labels.shape))
med_col = []
for entry in df.labels.values:
    entry = [names2[k] for k in entry]
    med_col.append(entry)

df['labels2'] = pd.Series(data=med_col)

# d=map(names2.get, df.labels.values)

def aggregate_values(series, agg_dict):
    """
    Combine multiple classes into a single class.
    Series object. If doing CV, you need to match indices.
    """
    lower_col = pd.Series(data=np.empty(series.shape))

    for k, v in agg_dict.items():
        lower_col[series.isin(v)] = k

    return lower_col

aggregate_values(df['labels2'], ldict)

lower_col = pd.Series(data=np.empty(df.labels.shape))


med_col.isin('1')

# for item in df.labels.values:
#     for cl in item:
#         for k, v in names.items():
#             if cl == v:
#                 cl = int(k)

#ranges = [(1,12), (12, 23), (23,26), (26,30), (30,35), (35,45)]

# dict_labs = {
#         '1': 'a',
#         '2': 2,
#         '3': 3,
#         '4': 4,
#         '5': 5,
#         '6': 6
#         }

# for rang in ranges:
#     for k, v in dict_labs.items():
#         v == [names[x] for x in [str(i) for i in range(rang)]]
# split_point = 299
# # path_col, label_col = read_patch(split_point)

# df, class_count = read_patch(split_point)

# label = labels[0] + '/'
# if label == 'Transitional woodland/shrub/':
#     label = 'Transitional woodland or shrub/'
# # class_rep.append(labels)

# # cl = []
# for elm in df['labels'].values:
#     for l in elm:
#         cl.append(l)
# lst = df['labels'].values
# d = list(itertools.chain.from_iterable(lst))

# classz = [x for l in df['labels'].values for x in l]


# # fl = ', '.join(cl)
# fl2 = shlex.split(fl)
# class_present = len(set(fl))

# # df, class_present, classes = read_patch(split_point)

# valz = list(df['labels'])
# fl = ', '.join(valz)

# fl=fl.split("' ")

# cp = set(fl)

# fl2 = shlex.split(fl)
# # class_present = len(set(fl2))

# fl = ', '.join(valz)
# fl2 = shlex.split(fl)
# class_present = len(set(fl2))

# # print('Patches sorted into classes...')
# # class_rep = set(df['labels'])

# # df["labels"] = df["labels"].apply(lambda x:x.split("'"))
# # df["labels"] = df["labels"].apply(lambda x:x.split("', "))
# # df["labels"] = df["labels"].apply(lambda x:re.split("', ", x))

# # class_count = len(set(class_rep))
# # print('Ready to train on {} RGB patches belonging to {}/{} classes.' \
# #   .format(len(patches), class_count, len(names)))
# # def split(delimiters, string, maxsplit=0):
# #     import re
# #     regexPattern = '|'.join(map(re.escape, delimiters))
# #     return re.split(regexPattern, string, maxsplit)

# delimiters = ", ", ", '"
# df["labels"] = df["labels"].apply(lambda x: split(delimiters, x))

# def split(delimiters, string, maxsplit=0):
#     import re
#     regexPattern = '|'.join(map(re.escape, delimiters))
#     return re.split(regexPattern, string, maxsplit)

# delimiters = ", ", ",'"

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


# patches, split_point = get_patches(patches)

# dst = '/DATA' if Server == True else '/data'

# cloud_patches = pd.read_csv(root_path + '{}/patches_with_cloud_and_shadow.csv'.format(dst), 'r', header=None)
# snow_patches = pd.read_csv(root_path + '{}/patches_with_seasonal_snow.csv'.format(dst), 'r', header=None)
# bad_patches = pd.concat([cloud_patches, snow_patches], axis=1)

# for patch in patches:
#     if any(patch in x for x in bad_patches.values):
#         patches.remove(patch)
#     else:
#         pass

# timeit.timeit(for patch in patches: /
#                 if patch in bad_patches.loc[:,:].values: /
#                     patches.remove(patch)
# # # bads = bad_patches.values

# patches[10] = 'S2B_MSIL2A_20171219T095409_35_52'

# print(len(bad_patches.iloc[:,1].values))
# print(len(patches))

# print(len(patches))

# for patch in patches:
#     if patch in bad_patches.loc[:,:].values:
#         patches.remove(patch)


# patch_series = pd.Series(patches)

# # if bad_patches.isin(patch_series)