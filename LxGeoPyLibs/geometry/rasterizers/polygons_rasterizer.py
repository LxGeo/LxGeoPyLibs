import pygeos
from LxGeoPyLibs.geometry.utils_rio import extents_to_profile
from LxGeoPyLibs.geometry.rasterizers import rasterize_from_profile
import rasterio as rio
from skimage import morphology
import numpy as np

def polygons_to_multiclass(geom_container, bounds, crs, contours_width=5, gsd=0.5):
    """
    Rasterize polygon geometries within specified bounds to three classes (empty, inner_map, contours_map).
    Args:
        geom_container: an iterable of geometries to burn.
        bounds: pygeos geometry or bounds tuple (minx , miny , maxx , maxy)
        crs: coordinate reference system of bounds.
        contours_width: size of the square used for inner polygon erosion
        gsd: raster pixel size in spatial metric
    """

    
    def binary_to_multiclass(x):
        """
        function used to transform building binary map to 3 classes (inner, countour, outer)
        """
        inner_map = morphology.erosion(x, morphology.square(contours_width))
        contour_map = x-inner_map
        background_map = np.ones_like(contour_map); background_map-=contour_map;background_map-=inner_map;
        return np.stack([background_map, inner_map, contour_map])

    if isinstance(bounds, pygeos.Geometry):
        bounds = pygeos.bounds(bounds)
    elif isinstance(bounds, (list, tuple)):
        pass
    else:
        raise Exception("bounds type unknown!")
    
    rasterization_profile = extents_to_profile(bounds, gsd=gsd, crs=crs, count=3, dtype=rio.uint8)
    if len(geom_container)==0:
        return np.zeros((rasterization_profile["count"], rasterization_profile["height"], rasterization_profile["width"]))
    polygon_rasterized = rasterize_from_profile(geom_container, rasterization_profile, 1)
    polygon_rasterized = binary_to_multiclass(polygon_rasterized)
    return polygon_rasterized

