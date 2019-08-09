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
#from skimage.io import imread
from tqdm import tqdm

from PIL import Image

from sklearn.preprocessing import MultiLabelBinarizer

st = time.time()
root_path = os.getcwd()

Server = False
Server = True

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

gsi_labels = [2, 12, 18, 23, 25, 41]

gsi_classes = [v for k,v in names.items() for l in gsi_labels if names[str(l)] == names[k]]

patches = [patches for patches in os.listdir(path_to_images)]

def get_patches(patches):

    st = time.time()
    valid_split = 0.3
    print('Unprocessed patch count: {}'.format(len(patches)))
    dst = '/DATA' if Server == True else '/data'
    cloud_patches = pd.read_csv(root_path + '{}/patches_with_cloud_and_shadow.csv'.format(dst), 'r', header=None)
    snow_patches = pd.read_csv(root_path + '{}/patches_with_seasonal_snow.csv'.format(dst), 'r', header=None)

    print('Purging cloud, shadow and snow patches...')

    patch_set = set(cloud_patches[0]).union(set(snow_patches[0]))
    patches = set(patches)
    patches = list(patches - patch_set)

    print('{} clean patches...'.format(len(patches)))
    split_point = int(len(patches) * valid_split)
    en = time.time()
    print('Purge took: {} secs '.format(float(en-st)))

    return patches, split_point

# patches, split_point = get_patches(patches)

def read_patch(bands = ['B02', 'B03', 'B04'], nodata=-9999):

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

    for i in tqdm(range(0, len(patches) - 1)):
        if Server:
            with open('/home/strathclyde/DATA/bigearth/dump/sample/{}/{}_labels_metadata.json' \
                      .format(patches[i], patches[i])) as js:
                meta = json.load(js)
        else:
            with open('data/sample/{}/{}_labels_metadata.json' \
                      .format(patches[i], patches[i])) as js:
                meta = json.load(js)

        labels = meta.get('labels')
        labels = [i for i in labels if i in gsi_classes]

        if labels == []:
            continue
        else:
            pass

        tifs = glob(path_to_images + '{}/*.tif'.format(patches[i]), recursive=True)
        band_tifs = [tif for tif in tifs for band in bands if band in tif]
        band_tifs.sort()
        files2rio = list(map(rio.open, band_tifs))

        # file2img = list(map(imread, band_tifs))
        # print(files2img)
        data = pl.stack(list(map(lambda x: x.read(1).astype(pl.int16), files2rio)))
        data = np.moveaxis(data, 0, 2)
        scipy.misc.toimage(data[...]).save(path_to_merge + patches[i] + '.png')
        path_col.append(path_to_merge + patches[i] + '.png')
        label_col.append(labels)

    d = {'path': path_col, 'labels': label_col}
    df = pd.DataFrame(d)
    names2 = {v: int(k) for k, v in names.items()}

    med_col = []
    for entry in df.labels.values:
        entry = [names2[k] for k in entry]
        med_col.append((entry))

    df['labels'] = pd.Series(data=med_col)
    lst = df['labels'].values
    classes = list(itertools.chain.from_iterable(lst))
    en = time.time()

    print('merged bands in {} sec'.format(float(en-st)))
    print('One hot encoding...')

    if i==6:
        print(df.labels)

    mlb = MultiLabelBinarizer()
    df = df.join(pd.DataFrame(mlb.fit_transform(df['labels']),
                          columns=mlb.classes_,
                          index=df.index))

    print('Dataframe ready')
    return df, len(set(classes))

# files2img = list(map(Image.fromarray, band_tifs))
# split_point = 299
# df, class_count = read_patch()

    # ldict = {
    #     1: [str(i) for i in range(1,12)],             # Artificial surfaces: 1 - 11
    #     2: [str(i) for i in range(12,23)],            # Agriculture: 12 - 22
    #     3: [str(i) for i in range(23,30)],            # Forest and vegetation: 23 - 30
    #     4: [str(i) for i in range(30,35)],            # Open space with little veg: 30-34
    #     5: [str(i) for i in range(35,45)]             # Water: 35 - 44
    # }

    # ent_df = []
    # for entry in df.l2.values:
    #     new_entry = []
    #     for elem in entry:
    #         for k,v in ldict.items():
    #             for val in v:
    #                 if elem == val:
    #                     elem = k
    #         new_entry.append(elem)
    #     ent_df.append(list(set(new_entry)))

    # df['labels'] = ent_df
    # df = df.drop(['l2'], axis=1)

    # ldict = {
    #     1: [str(i) for i in range(1,12)],             # Artificial surfaces: 1 - 11
    #     2: [str(i) for i in range(12,23)],            # Agriculture: 12 - 22
    #     3: [str(i) for i in range(23,26)],            # Forest: 23, 24, 25
    #     4: [str(i) for i in range(26,30)],            # Vegetation: 26-29
    #     5: [str(i) for i in range(30,35)],            # Open space with little veg: 30-34
    #     6: [str(i) for i in range(35,45)]             # Water: 35 - 44
    # }

# split_point = 299
# print(class_count)
# # cldict = {
# #         1: [names[x] for x in [str(i) for i in range(1,12)]],                   # Artificial surfaces: 1 - 11
# #         2: [names[x] for x in [str(i) for i in range(12,23)]],                  # Agriculture: 12 - 22
# #         3: [names[x] for x in [str(i) for i in range(23,26)]],                    # Forest: 23, 24, 25
# #         4: [names[x] for x in [str(i) for i in range(26,30)]],                # Vegetation: 26-29
# #         5: [names[x] for x in [str(i) for i in range(30,35)]],            # Open space with little veg: 30-34
# #         6: [names[x] for x in [str(i) for i in range(35,45)]]               # Water: 35 - 44
# #     }

# ldict = {
#         1: [str(i) for i in range(1,12)],                   # Artificial surfaces: 1 - 11
#         2: [str(i) for i in range(12,23)],                  # Agriculture: 12 - 22
#         3: [str(i) for i in range(23,26)],                    # Forest: 23, 24, 25
#         4: [str(i) for i in range(26,30)],                # Vegetation: 26-29
#         5: [str(i) for i in range(30,35)],            # Open space with little veg: 30-34
#         6: [str(i) for i in range(35,45)]               # Water: 35 - 44
#     }

# #

# names2 = {v: k for k, v in names.items()}


# med_col = []
# for entry in df.labels.values:
#     entry = [names2[k] for k in entry]
#     med_col.append((entry))

# df['l2'] = pd.Series(data=med_col)


# ent_df = []
# for entry in df.l2.values:
#     new_entry = []
#     for elem in entry:
#         for k,v in ldict.items():
#             for val in v:
#                 if elem == val:
#                     elem = k
#         new_entry.append(elem)
#     ent_df.append(new_entry)

# df['l3'] = ent_df
# df = df.drop(['labels'], axis=1)
# df = df.drop(['l2'], axis=1)
# df['labels'] = df['l3']
# df = df.drop(['l3'], axis=1)
# df.head()


# d=map(names2.get, df.labels.values)

# def aggregate_values(series, agg_dict):
#     """
#     Combine multiple classes into a single class.
#     Series object. If doing CV, you need to match indices.
#     """
#     lower_col = pd.Series(data=np.zeros(series.shape))
#     for k, v in agg_dict.items():
#         lower_col[series.isin(v)] = k

#     return lower_col

# aggregate_values(df.l3, ldict)

# lower_col = pd.Series(data=np.empty(df.labels.shape))

# df.l3 = df.l3.apply(lambda x: int(x).split('[]'))

# el = []
# for elem in df.l3.values:
#     elem.astype(int)
#     el.append(elem)
# el.append(eleml)

# for i in range(0, len(df.l3.values) - 1):
#     ent = df['l3'].iloc[i].astype(int)

    # # Need to convert df.labels to names.key instead of names.value

# ldict2 = {
#         1: [1,2,3],
#         2: [5]}
# for k in names2:
#     names2[k] = int(names2[k])

# df.l2 = df.l2.map(names2)

# df["l3"] = df["l2"].apply(lambda x: str(x).strip('[]'))

# df['labels'] = df['labels'].map(pd.Series(names2))
# df.labels
# df['labels'].replace(names2)
# df = df.replace(df.labels, names)

# df.replace({'labels': names2})

# df['labels'].put(names2.keys(), names2.values())

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