import math
import pyproj
from shapely.geometry import Polygon, mapping
from shapely.ops import transform
from functools import partial


def utm_zone_code(lon, lat):
    """
    Returns the UTM zone code for the input longitude and latitude.
    """
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if lat >= 0:
        epsg_code = '326' + utm_band.zfill(2)
    else:
        epsg_code = '327' + utm_band.zfill(2)
    
    return int(epsg_code)


def is_utm(crs):
    # Retrieve the coordinate operation string
    coord_operation = crs.coordinate_operation    
    # Return whether it contains 'UTM'
    return 'UTM' in str(coord_operation)


def reproject_geom(geom, src_crs, dst_crs):
    """
    Reprojects a shapely geometry from a source coordinate refrence system to target one.
    Args:
        geom: shapely geometry
        src_crs: pyproj.CRS 
        dst_crs: pyproj.CRS 
    """
    project = partial(
        pyproj.transform,
        pyproj.Proj(src_crs),
        pyproj.Proj(dst_crs)
    )
    return transform(project, geom)



