import rasterio as rio
from rasterio.merge import merge
from rasterio.mask import mask
from geopandas import GeoDataFrame, read_file
import pylab as pl
from glob import glob
import os
from subprocess import check_output
from .io import write_raster
from rasterio.warp import reproject, Resampling, calculate_default_transform

def copy_projection(
    copy_tif,
    src_path='data/Corine_Orig_10m_OS_AoI1.tif', 
    dst_path='data/Corine_S2_Proj.tif'):
    """
    Copy the projection of a geotiff to another one.
    args:
        copy_tif <- either an opened dataset or a path to a geotiff from which to copy the projection
        src_path <- path to dataset to copy values from
        dst_path <- path to place where the copycat tif will be written
    """
    assert isinstance(copy_tif, rio.DatasetReader) or isinstance(copy_tif, str)
    assert isinstance(src_path, str)
    
    if isinstance(copy_tif, rio.DatasetReader):
        dst_crs = copy_tif.crs
    elif isinstance(copy_tif, str):
        copy_tif = rio.open(copy_tif)
        dst_crs = copy_tif.crs
        

    with rio.open(src_path) as src:
        assert dst_crs != src.crs
        c_trans, c_width, c_height = calculate_default_transform(
            src.crs,
            dst_crs, 
            src.width, 
            src.height,
            *src.bounds)
        
        meta = src.meta.copy()
        meta.update({
            'crs': dst_crs,
            'transform': copy_tif.transform,
            'width': copy_tif.width,
            'height': copy_tif.height
        })

        with rio.open(dst_path, 'w', **meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=copy_tif.transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )

root_path = check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode()


def merge_bands(safe_id, res='10', bands = ['B02', 'B03', 'B04', 'B08']):
    """
    Combines several single band rasters into a single multiband raster
    args:
        safe_id <- just the name of directory ending in SAFE.
        res <- resolution to use. We have 10, 20, and 60.
        bands <- the bands to use.
    """
    files = glob(root_path + '/data/{}/**/R{}m/*_B*'.format(safe_id, res), recursive=True)
    files.sort()
    tifs = list(map(rio.open, files))
    data = pl.stack(list(map(lambda x: x.read(1).astype(pl.int16), tifs)))
    profile = tifs[0].profile
    return data, profile

def merge_scene_bands(date, outpath, res=10, bands = ['B02', 'B03', 'B04', 'B08']):
    """
    Same as merge_bands() but only in cases where your AOI intersects multiple scenes.
    """
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

def mask_raster(shp, merge_path, outpath):
    """
    Clips a raster based on a shapefile.
    args:
        shp <- a dataframe from loading in a shapefile object or a path to a shapefile
        merge_path <- path to combined multiband raster or the object if you had loaded it in beforehand
        outpath <- path to where to put the output
    """
    assert isinstance(outpath, str)
    assert isinstance(merge_path, str) or isinstance(merge_path, rio.DatasetReader)
    
    if isinstance(shp, str):
        shp = read_file(shp)
    elif isinstance(shp, GeoDataFrame):
        shp = shp

    if isinstance(merge_path, str):
        m_src = rio.open(merge_path)
    else:
        m_src = merge_path
    
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
    """
    Remove NA, inf, and other nasty values that are results of division by zero
    from calculating indices. Multiplies by 100 to discretise values.
    
    I think you'll want to experiment with discrete/continuous values for the index.
    You could have better clustering with floating points than integers.
    
    args:
        index <- array from calculating index
        
    out:
        cleaned array
    """
    index[(index == pl.inf) | (index == -pl.inf)] = 0
    index = (100 * index)
    return index.fillna(0).astype(pl.int16)

def calc_ndvi(df):
    """
    Calcuates the Normalised Difference Vegetation Index from S2 NIR and RED bands
    """
    index = (df['B08'] - df['B04']) / (df['B08'] + df['B04'])
    return clean_index(index)


def calc_savi(df, L = 0.5):
    """
    Calculates the Soil-adjusted Vegetation Index.
    """
    index = (df['B08'] - df['B04']) * (1 + L) / (df['B08'] + df['B04'] + L) 
    return clean_index(index)


def calc_evi(df, G = 2.5, C1 = 6., C2 = 7.5, L_evi = 1.):
    """
    Calculates the Enhanced Vegetation Index
    """
    index = G * ((df['B08'] - df['B04']) / ((df['B08'] + (C1 * df['B04']) - (C2 * df['B02'])) + L_evi))
    return clean_index(index)


def calc_ndwi(df):
    """
    Calculates the Normalised Difference Water Index. May not work too well in some cases.
    """
    index = (df['B03'] - df['B08']) / (df['B03'] + df['B08'])
    return clean_index(index)

def calc_indices(df):
    """
    Nicely add the indices to the dataframe. May need extending if you find more indices.
    """
    df['savi'] = calc_savi(df)
    df['ndwi'] = calc_ndwi(df)
    df['evi'] = calc_evi(df)
    df['ndvi'] = calc_ndvi(df)
    return df