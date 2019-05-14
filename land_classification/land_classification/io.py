import rasterio as rio
from glob import glob
import pylab as pl

def write_raster(name, data, profile, nodata=-9999):
    """
    Wrapper for writing multiband rasters.
    
    Input arguments:
        name <- Filename to be written.
        data <- Numpy ndarray to be written.
        tif <- Original geotiff that was read.
        transform <- Modified transform of raster.
    """
    profile.update({
        "driver": "GTiff",
        "count": data.shape[0],
        "height": data.shape[1],
        "width": data.shape[2],
        "nodata": nodata,
        "dtype": data.dtype,
        "crs": 'EPSG:'
    })
    
    print("Writing raster {} with {} bands.".format(name, data.shape[0]))
    
    with rio.open(name, "w", **profile) as dst:
        for i in range(len(data)):
            print("Writing band {} of {}".format(i+1, data.shape[0]))
            dst.write(data[i], i+1)
            
    print("File {} written.".format(name))