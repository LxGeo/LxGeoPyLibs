import geopandas as gpd
from shapely.geometry import Polygon, base
import numpy as np

def make_grid(geom_container, x_step, y_step):
    """
    Create a grid covering the extents of geometry container.
    Args:
        geom_container: shapely geometry || geopandas dataframe
        x_step, y_step: float
    Returns:
        list of shapely polygons constructing the grid
    """
    if isinstance(geom_container, gpd.GeoDataFrame):
        total_bounds = geom_container.total_bounds
    elif isinstance(geom_container, base.BaseGeometry):
        total_bounds = geom_container.bounds
    else:
        raise Exception("Cannot make grid for geometry container of type {}".format(type(geom_container)))
    
    xmin, ymin, xmax, ymax = total_bounds

    cols = list(np.arange(xmin, xmax + x_step, x_step))
    rows = list(np.arange(ymin, ymax + y_step, y_step))

    polygons = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(Polygon([(x,y), (x+x_step, y), (x+x_step, y+y_step), (x, y+y_step)]))
    
    return polygons
