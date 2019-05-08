import rasterio as rio
from fiona.crs import from_epsg
import pylab as pl
from shapely.geometry import Point
from pyproj import Proj, transform
import geopandas as gpd
import pandas as pd


class PointExtractor:
    """
    Extract a single point from a polygon shapefile.
    """
    def __init__(self, shp):
        self.shp = shp
        self.p = None

    def get(self):
        """
        Retrieve a x/y pair from shapefile.
        """
        poly = self._sample()

        (minx, miny, maxx, maxy) = poly.bounds
        while True:
            self.p = Point(pl.uniform(minx, maxx), pl.uniform(miny, maxy))
            if poly.contains(self.p):
                return self._reproject_value()

    def get_n(self, n_points):
        """
        Retrieve n_points x/y pairs
        """
        df = gpd.GeoDataFrame()
        points = pl.zeros((n_points, 2))
        for i in range(n_points):
            p_tmp = self.get()
            points[i] = [p_tmp[1], p_tmp[0]]
        df['Lat'] = points[:, 0]
        df['Lon'] = points[:, 1]
        df['Val'] = 0
        geometry = [Point(xy) for xy in zip(df.Lon, df.Lat)]
        df['geometry'] = geometry
        df.crs = from_epsg(4326)
        return df

    def _sample(self):
        """
        Samples a random row (polygon) in the shapefile
        """
        return self.shp.sample(1)['geometry'].values[0]

    def _reproject_value(self):
        """
        Convert from original shapefile projection to WGS84
        """
        in_proj = Proj(init=self.shp.crs['init'])
        Proj(init=self.shp.crs['init'])
        #out_proj = in_proj
        out_proj = Proj(init='epsg:4326')
        return transform(in_proj, out_proj, self.p.x, self.p.y)

def sample_raster(df, path, bands=['B02', 'B03', 'B04', 'B08']):
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

    values = []
    for i, j in zip(*tif.index(df['geometry'].x, df['geometry'].y)):
        values.append(arr[:, i, j])
        
    new_df = pd.DataFrame(data=values, columns=bands)
    df[bands] = new_df[bands]
    
    return df
