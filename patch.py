# import geopandas as gpd
# import rasterio as rio
from skimage.io import imread

# from rasterio import features
# from rasterio.merge import merge
# from rasterio.plot import show
# import pylab as pl
# from fiona.crs import from_epsg
# from shapely.geometry import box as geobox
# from shapely.geometry import shape
# import json
# import geojson
import numpy as np
# import pandas as pd
# import os
# from glob import glob
# from subprocess import check_output
# import matplotlib.pyplot as plt
# from keras.models import Sequential
# import land_classification as lc
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.regularizers import l2
# from keras import optimizers

image = np.array(imread('data/segment/TCI_clipped.tif'), dtype=float)

_, num_cols_prepad, _ = image.shape

