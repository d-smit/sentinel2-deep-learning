import rasterio as rio
from rasterio.merge import merge
from rasterio.mask import mask
import pylab as pl
from glob import glob
import os
from subprocess import check_output
from .io import write_raster


root_path = check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode()


def merge_bands(safe_id, res='10', bands = ['B02', 'B03', 'B04', 'B08']):
    files = glob(root_path + '/data/{}/**/R{}m/*_B*'.format(safe_id, res), recursive=True)
    files.sort()
    tifs = list(map(rio.open, files))
    data = pl.stack(list(map(lambda x: x.read(1).astype(pl.int16), tifs)))
    profile = tifs[0].profile
    return data, profile

def merge_scene_bands(date, outpath, res=10, bands = ['B02', 'B03', 'B04', 'B08']):
    scenes = glob(root_path + '/data/archives/*L2A*{}*/**/R{}m/*B*'.format(date), recursive=True)
    scenes.sort()
    toconc = []
    for band in bands:
        tomerge = []
        for scene in scenes:
            if band in scene:
                tif = rio.open(scene)
                tomerge.append(tif)
        merge_arr, merge_trans = merge(tomerge)
        toconc.append(merge_arr)
    merged_stack = pl.vstack(toconc).astype(pl.int16)
    write_raster(outpath, merged_stack, tif, merge_trans)

def mask_raster(shp, mask_path, outpath):
    m_src = rio.open(mask_path)
    shp = shp.to_crs(m_src.crs)
    mask_arr, mask_trans = mask(m_src, 
                                shapes=shp['geometry'], 
                                crop=True, 
                                nodata=-9999,
                                all_touched=True)
    mask_arr[mask_arr == -32768] = -9999
    mask_arr = mask_arr.astype(pl.int16)
    profile = m_src.profile
    profile['transform'] = mask_trans

    write_raster(outpath, mask_arr, profile)
    
def clean_index(index):
    index[(index == pl.inf) | (index == -pl.inf)] = 0
    index = (100 * index)
    return index.fillna(0).astype(pl.int16)

def calc_ndvi(df):
    index = (df['B08'] - df['B04']) / (df['B08'] + df['B04'])
    return clean_index(index)


def calc_savi(df, L = 0.5):
    index = (df['B08'] - df['B04']) * (1 + L) / (df['B08'] + df['B04'] + L) 
    return clean_index(index)


def calc_evi(df, G = 2.5, C1 = 6., C2 = 7.5, L_evi = 1.):
    index = G * ((df['B08'] - df['B04']) / ((df['B08'] + (C1 * df['B04']) - (C2 * df['B02'])) + L_evi))
    return clean_index(index)


def calc_ndwi(df):
    index = (df['B03'] - df['B08']) / (df['B03'] + df['B08'])
    return clean_index(index)

def calc_indices(df):
    df['savi'] = calc_savi(df)
    df['ndwi'] = calc_ndwi(df)
    df['evi'] = calc_evi(df)
    df['ndvi'] = calc_ndvi(df)
    return df