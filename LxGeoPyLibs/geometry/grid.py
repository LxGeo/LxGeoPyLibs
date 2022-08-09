from cgitb import small
import geopandas as gpd
from shapely.geometry import base
import pygeos
import numpy as np

def make_grid(geom_container, x_step, y_step, x_size, y_size, filter_predicate=lambda x:True):
    """
    Create a grid covering the extents of geometry container.
    Args:
        geom_container: shapely geometry || geopandas dataframe
        x_step, y_step: float
    Returns:
        list of shapely polygons constructing the grid
    """
    if isinstance(geom_container, gpd.GeoDataFrame):
        geom_container = pygeos.box(*geom_container.total_bounds)
    elif isinstance(geom_container, base.BaseGeometry):
        geom_container = pygeos.from_shapely(geom_container)
    elif isinstance(geom_container, pygeos.Geometry):
        geom_container = geom_container
    else:
        raise Exception("Cannot make grid for geometry container of type {}".format(type(geom_container)))    
    xmin, ymin, xmax, ymax = pygeos.bounds(geom_container)
    cols = list(np.arange(xmin, xmax+x_step, x_step))
    rows = list(np.arange(ymin, ymax+y_step, y_step))
    polygons = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            small_window = pygeos.linearrings([(x,y), (x+x_size, y), (x+x_size, y+y_size), (x, y+y_size)])
            if filter_predicate(small_window):
                polygons.append(small_window)    
    polygons = pygeos.polygons(polygons)
    return polygons
