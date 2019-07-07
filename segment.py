from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import geopandas as gpd
import tensorflow as tf
import numpy as np
import pandas as pd
import rasterio as rio
import pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns

import land_classification as lc

from rasterio import features
from rasterstats import zonal_stats
from fiona.crs import from_epsg
from shapely.geometry import box as geobox
from shapely.geometry import shape
import json
import geojson

from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb
import tifffile as tiff
import scipy.misc

def create_segment_polygon(tif):
    data = tif.read()
    print("Segmenting S2 image using quickshift")
    segments = felzenszwalb(np.moveaxis(data, 0, -1),  scale=100, sigma=0.5, min_size=50)
    print("Extracting shapes")
    shapes = list(features.shapes(segments.astype(np.int32), transform=tif.transform))
    print("Creating GeoDataFrame from polygon segments.")
    seg_gdf = gpd.GeoDataFrame.from_dict(shapes)
    seg_gdf['geometry'] = seg_gdf[0].apply(lambda x: shape(geojson.loads(json.dumps(x))))
    seg_gdf.crs = tif.crs
    print("Segment CRS: {}".format(seg_gdf.crs))
    return seg_gdf.drop([0, 1], axis=1), segments, shapes

# Background image

def plot_segments(segments):
    print('Plotting segments over scene render...')

    image_to_plot = rio.open('data/segment/masked_image_render.tif')
    data = image_to_plot.read()

    plot_data = data[0:3]  # mark_boundaries needs [width, height, 3] format

    scipy.misc.toimage(plot_data, cmin=0.0).save('masked_plot.tif')
    image_plot = tiff.imread('masked_plot.tif')
    fig, ax = plt.subplots(1, 1, figsize=(25, 25), sharex=True, sharey=True)
    ax.imshow(mark_boundaries(image_plot, segments))
    plt.tight_layout()
    plt.savefig('masked_segmented.tif')
    plt.show()

# Getting segment IDs from segments_df

def get_zones_and_dists(df):
    print('Getting zonal stats for segments over scene...')

    ''' Getting zonal stats for each segment and calculating 
        mean distributions. '''

    segment_stats = zonal_stats(df, 'data/swindon/masked.tif',
                                stats = 'count min mean max median')

    zones_df = pd.DataFrame(segment_stats)
    zones_df = zones_df.dropna()
    zones_df['zone_id'] = np.nan

    zone_means = zones_df['mean']

    dists = [10, 25, 50, 75, 90]

    dist_values = []

    for i in range(0, len(dists)):
        d = np.percentile(zone_means, dists[i])
        dist_values.append(d)

    return zones_df, tuple(dist_values)

def tag_zones(df, dv):
    print('Tagging zones...')

    ''' Giving each segment a number between 1-5 based on their 
        similarity to the distribution percentiles. '''

    for idx, row in df.iterrows():
        if (df.at[idx, 'mean']) < 1.2 * dv[0]:
            df.at[idx, 'zone_id'] = 1

        elif 1.2 * dv[0] < (df.at[idx, 'mean']) < 1.1 * dv[1]:
            df.at[idx, 'zone_id'] = 2

        elif 1.1 * dv[1] < (df.at[idx, 'mean']) < 1.1 * dv[2]:
            df.at[idx, 'zone_id'] = 3

        elif 1.1 * dv[2] < (df.at[idx, 'mean']) < 1.2 * dv[3]:
            df.at[idx, 'zone_id'] = 4

        elif 1.2 * dv[3] < (df.at[idx, 'mean']):
            df.at[idx, 'zone_id'] = 5

    df = df.dropna()

    return df

def match_segment_id(pixel_df, poly_df):
    print('Matching pixels with their segment ID...')

    ''' Parsing through the extracted points and matching
        each pixel with the segment ID of the segment
        that contains it. '''

    for i, row in enumerate(pixel_df.itertuples(), 0):
        point = pixel_df.at[i, 'geometry']

        for j in range(len(poly_df)):
              poly = poly_df.iat[j, 0]

              if poly.contains(point):
                  pixel_df.at[i, 'segment_id'] = poly_df.iat[j, 1]
              else:
                  pass

    return pixel_df    # trying to make this faster

# tif = rio.open('data/swindon/masked.tif')

# segment_df, segments, shapes = create_segment_polygon(tif)
# #zones_df, dist_values = get_zones_and_dists(segment_df)
# zones_df = tag_zones(zones_df, dist_values)

# segment_df['polygon id'] = zones_df['zone_id']

# # segment_df = segment_df.dropna()

# # Now need to add zone_id column to extracted pixels dataframe

# pe = lc.PointExtractor(aoi)

# points_df = pe.get_n(5000)

# bands = ['B02', 'B03', 'B04', 'B08']

# points_df = lc.sample_raster(points_df, 'data/Corine_S2_Proj_2.tif', bands=['labels'])
# points_df['segment_id'] = np.nan
# #points_df = match_segment_id(points_df, segment_df)
# points_df = lc.sample_raster(points_df, 'data/swindon/masked.tif', bands=bands)

# clean_df = lc.remove_outliers(points_df, bands=bands, indices=False)
# clean_df = lc.calc_indices(clean_df)

# class_cols = 'labels_1'
 
# predictors = ['B02_1','B03_1',
#               'B04_1','B08_1',
#               'ndwi',
#               'segment_id']

# clean_df = clean_df.drop(['savi', 'evi', 'ndvi'], axis=1)

# X = clean_df[predictors]
# X = X.values
# y = clean_df[class_cols]
# y = y.values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# X_train = preprocessing.scale(X_train)
# X_test = preprocessing.scale(X_test)

# preds = len(predictors)
# labs = len(list(clean_df[class_cols].unique()))

# input_num_units = preds
# hidden1_num_units = 200
# hidden2_num_units = 200
# hidden3_num_units = 200
# hidden4_num_units = 200
# output_num_units = labs

# model = Sequential([
#     Dense(output_dim=hidden1_num_units,
#           input_dim=input_num_units,
#           kernel_regularizer=l2(0.0001),
#           activation='relu'),
#     Dropout(0.2),
#     Dense(output_dim=hidden2_num_units,
#           input_dim=hidden1_num_units,
#           kernel_regularizer=l2(0.0001),
#           activation='relu'),
#     Dropout(0.2),
#     Dense(output_dim=hidden3_num_units,
#           input_dim=hidden2_num_units,
#           kernel_regularizer=l2(0.0001),
#           activation='relu'),
#     Dropout(0.1),
#     Dense(output_dim=hidden4_num_units,
#           input_dim=hidden3_num_units,
#           kernel_regularizer=l2(0.0001),
#           activation='relu'),
#     Dropout(0.1),
#     Dense(output_dim=(max(clean_df[class_cols])+1),
#           input_dim=hidden4_num_units, 
#           activation='softmax'),
#  ])
    
# model.summary()

# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# # Compile model

# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])

# history=model.fit(X_train, 
#           y_train,
#           epochs=100, 
#           batch_size=100, 
#           validation_split = 0.2,
#           verbose=1,
#           )

# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='lower right')
# plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.show()

# # Model evaluation with test data set
# # Prediction at test data set

# y_pred = model.predict(X_test)
# score = model.evaluate(X_test, y_test, batch_size=100, verbose=1)

# # score_2 = model.score(X_test, y_test)
# print(score)
# print("Baseline Error: %.2f%%" % (100-score[1]*100))

# Load and prepare the dataset to predict on

# mask_src = tif
# profile = mask_src.profile
# data = mask_src.read(list(pl.arange(mask_src.count) + 1))
# gdf = lc.create_raster_df(data, bands=bands)
# gdf = lc.calc_indices(gdf)

# cm = confusion_matrix(y_pred, y_test)
# f, ax = pl.subplots(1, figsize=(20, 20))
# sns.heatmap(ax=ax,
#             data=cm, 
#             annot=True, 
#             fmt='g',
#             cmap='pink',
#             linewidths=0.5, 
#             cbar=False)
# ax.set_ylabel('Predicted')
# ax.set_xlabel('True')
# ax.set_title('Confusion matrix for Corine Level-2 Groups')
# # pl.xticks(pl.arange(len(y_test.unique()))+0.5, plot_names, rotation=45)
# # pl.yticks(pl.arange(len(y_test.unique()))+0.5, plot_names, rotation=45)
# f.show()
# f.savefig('outputs/cv_{}.png'.format('segmented_DNN'))

