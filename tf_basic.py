import geopandas as gpd
import rasterio as rio
from rasterio.merge import merge
from rasterio.plot import show
import pylab as pl
import json
import pandas as pd
import os
from glob import glob
from subprocess import check_output
import matplotlib.pyplot as plt
from keras.models import Sequential
from fiona.crs import from_epsg
from shapely.geometry import box as geobox
import land_classification as lc
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Defining Swindon area of interest

aoi_geo = geobox(-2.29, 51.51, -1.71, 51.61)
aoi = gpd.GeoDataFrame([], geometry=[aoi_geo])
aoi.crs = from_epsg(4326)
aoi.to_file('data/aoi.geojson', driver='GeoJSON')

# Getting land-cover classes 

with open('data/labels.json') as jf:
    names = json.load(jf)
    
root_path = check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode()

# Reading and merging DEM data

#def merge_dem():
#
#    files = glob(root_path + '/data/Ancillary/swindon*', recursive=True)
#    files.sort()
#    tifs = list(map(rio.open, files))
#    dem_data = pl.stack(list(map(lambda x: x.read(1).astype(pl.int16), tifs)))
#    dem_profile = tifs[0].profile
#    return dem_data, dem_profile
#
#dem_data, dem_profile = merge_dem()

# Reading and merging band data

s2_band = 'S2A.SAFE'
data, profile = lc.merge_bands(s2_band, res='10')

# Writing and masking band raster

lc.write_raster('data/swindon/merged.tif', data, profile)
lc.mask_raster(aoi, 'data/swindon/merged.tif', 'data/swindon/masked.tif')

## Writing and masking DEM raster 
#
#lc.write_raster('data/swindon/merged_dem.tif', dem_data, dem_profile)    
#lc.mask_raster(aoi, 'data/swindon/merged_dem.tif', 'data/swindon/masked_dem.tif')
#
## Making mosaic of both
#
#masks = os.path.join(root_path, 'data/swindon/masked*.tif')
#
#bands_dems = glob(masks)
#
#band_ancillary_mosaic = []
#
#for tif in bands_dems:
#    src = rio.open(tif)
#    band_ancillary_mosaic.append(src)
#    
#mosaic, out_trans = merge(band_ancillary_mosaic, indexes=range(12))

pe = lc.PointExtractor(aoi)
 
points_df = pe.get_n(3000)

bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08']
 
def sample_raster(df, path, bands=['B02', 'B03', 'B04', 'B08'], buffer=5):

    assert isinstance(path, str) or isinstance(path, rio.DatasetReader)
    if isinstance(path, str):
        tif = rio.open(path)
    else:
        tif = path

    df = df.to_crs(from_epsg(tif.crs.to_epsg()))

    if tif.count == 1:
        arr = tif.read()
    else:
        arr = tif.read(list(pl.arange(tif.count) + 1))
    print(arr)
    values = []
    for i, j in zip(*tif.index(df['geometry'].x, df['geometry'].y)):
        values.append(arr[:, i-buffer:(i+1)+buffer, j-buffer:(j+1)+buffer])
       
    cols = [band + '_' + str(v+1) for band in bands for v in range(values[0].shape[1] * values[0].shape[2])]
    new_df = pd.DataFrame(data=list(map(lambda x: x.flatten(), values)), columns=cols)
    df[new_df.columns] = new_df
    return df

points_df = sample_raster(points_df, 'data/Corine_S2_Proj_2.tif', bands=['labels'])
points_df.iloc[1,:]

points_df = sample_raster(points_df, 'data/swindon/masked.tif', bands=bands)
points_df.iloc[1,:]


points_df = lc.sample_raster(points_df, 'data/Corine_S2_Proj_2.tif', bands=['labels'])
points_df = lc.sample_raster(points_df, 'data/swindon/masked.tif', bands=bands)
 
clean_df = lc.remove_outliers(points_df, bands=bands, indices=False)
clean_df = lc.calc_indices(clean_df)
 
class_cols = 'labels_1'
 
predictors = ['B01_1', 'B02_1', 'B03_1', 'B04_1', 'B05_1', 'B06_1', 'B07_1', 'B08_1', 'ndwi']

clean_df = clean_df.drop(['savi'], axis=1)
clean_df = clean_df.drop(['evi'], axis=1)
clean_df = clean_df.drop(['ndvi'], axis=1)

X = clean_df[predictors]
X = X.values
y = clean_df[class_cols]
y = y.values
 
#mask_src = rio.open('data/masked.tif')
# 
#profile = mask_src.profile
#data = mask_src.read(list(pl.arange(mask_src.count) + 1))
#gdf = lc.create_raster_df(data, bands=predictors)
#gdf = lc.calc_indices(gdf)
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

preds = len(predictors);preds
labs = len(list(clean_df[class_cols].unique()))

input_num_units = preds
hidden1_num_units = 200
hidden2_num_units = 200
hidden3_num_units = 200
hidden4_num_units = 200
output_num_units = labs

model = Sequential([
    Dense(output_dim=hidden1_num_units,
          input_dim=input_num_units,
          kernel_regularizer=l2(0.0001),
          activation='relu'),
    Dropout(0.2),
    Dense(output_dim=hidden2_num_units,
          input_dim=hidden1_num_units,
          kernel_regularizer=l2(0.0001),
          activation='relu'),
    Dropout(0.2),
    Dense(output_dim=hidden3_num_units,
          input_dim=hidden2_num_units,
          kernel_regularizer=l2(0.0001),
          activation='relu'),
    Dropout(0.1),
    Dense(output_dim=hidden4_num_units,
          input_dim=hidden3_num_units,
          kernel_regularizer=l2(0.0001),
          activation='relu'),
    Dropout(0.1),
    Dense(output_dim=(max(clean_df[class_cols])+1),
          input_dim=hidden4_num_units, 
          activation='softmax'),
 ])
    
model.summary()

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Compile model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history=model.fit(X_train, 
          y_train,
          epochs=100, 
          batch_size=100, 
          validation_split = 0.2,
          verbose=1,
          )

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# Model evaluation with test data set 
# Prediction at test data set
y_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test, batch_size=100, verbose=1)
print(score)
print("Baseline Error: %.2f%%" % (100-score[1]*100))


# Performance on other gdf

