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
    elif isinstance(bounds, (list, tuple, np.ndarray)):
        pass
    else:
        raise Exception("bounds type unknown!")
    
    rasterization_profile = extents_to_profile(bounds, gsd=gsd, crs=crs, count=3, dtype=rio.uint8)
    if len(geom_container)==0:
        return np.zeros((rasterization_profile["count"], rasterization_profile["height"], rasterization_profile["width"]))
    polygon_rasterized = rasterize_from_profile(geom_container, rasterization_profile, 1)
    polygon_rasterized = binary_to_multiclass(polygon_rasterized)
    return polygon_rasterized

def polygons_to_multiclass2(gdf, bounds, crs, contours_width=3, gsd=0.5):
    if isinstance(bounds, pygeos.Geometry):
        bounds = pygeos.bounds(bounds)
    elif isinstance(bounds, (list, tuple, np.ndarray)):
        pass
    else:
        raise Exception("bounds type unknown!")
    
    rasterization_profile = extents_to_profile(bounds, gsd=gsd, crs=crs, count=3, dtype=rio.uint8)
    if (gdf.empty):
        proba_map = np.zeros((rasterization_profile["count"], rasterization_profile["height"], rasterization_profile["width"]))
        proba_map[0]=1
    else:
        contour_rasterized = rasterize_from_profile(gdf.geometry.boundary, rasterization_profile, 1)
        contour_rasterized=morphology.dilation(contour_rasterized, morphology.square(contours_width))
        polygon_rasterized = rasterize_from_profile(gdf.geometry, rasterization_profile, 1)
        polygon_rasterized = np.maximum(polygon_rasterized.astype(int)-contour_rasterized.astype(int), 0)
        background_map = np.maximum( np.ones_like(contour_rasterized).astype(int)-polygon_rasterized.astype(int)-contour_rasterized.astype(int),0 )

        proba_map = np.stack([background_map, polygon_rasterized, contour_rasterized])
    return proba_map


def polygonsWA_to_displacment_map(gdf, bounds, crs, gsd=0.5, disp_x_column_name="disp_x",disp_y_column_name="disp_y", weight_column_name="hrel"):
    """
    """
    if isinstance(bounds, pygeos.Geometry):
        bounds = pygeos.bounds(bounds)
    elif isinstance(bounds, (list, tuple)):
        pass
    else:
        raise Exception("bounds type unknown!")
    
    flow_profile = extents_to_profile(bounds, crs=crs, count=3, dtype=np.float32, gsd=gsd)
    if gdf.empty:
        return np.zeros((flow_profile["count"], flow_profile["height"], flow_profile["width"]))

    dispx = rasterize_from_profile(gdf.geometry, flow_profile, gdf[disp_x_column_name].values / gsd)
    dispy = rasterize_from_profile(gdf.geometry, flow_profile, gdf[disp_y_column_name].values / -gsd)
    weight = rasterize_from_profile(gdf.geometry, flow_profile, gdf[weight_column_name].values, background_fill=1)
    weighted_flow = np.stack([dispx, dispy, weight])
    return weighted_flow