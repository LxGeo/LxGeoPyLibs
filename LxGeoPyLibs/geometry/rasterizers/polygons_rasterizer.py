import pygeos
from LxGeoPyLibs.geometry.utils_rio import extents_to_profile
from LxGeoPyLibs.geometry.rasterizers import rasterize_from_profile
import rasterio as rio
from skimage import morphology
import numpy as np
from scipy.ndimage import gaussian_filter

def polygons_to_multiclass(gdf, bounds, crs, gsd=0.5, contours_width=2, inner_width=2, blur=None):
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
        
        get_morph_op = lambda x: morphology.erosion if x < 0 else morphology.dilation if x > 0 else lambda y, *args: y
        
        c_morphology_op = get_morph_op(inner_width)
        inner_map = c_morphology_op(x, morphology.square(abs(inner_width)))
        
        c_morphology_op = get_morph_op(contours_width)
        contour_map = c_morphology_op(inner_map, morphology.square(abs(contours_width))) - inner_map
                
        background_map = np.ones_like(contour_map); background_map-=contour_map;background_map-=inner_map;
        return np.stack([background_map, inner_map, contour_map])

    if isinstance(bounds, pygeos.Geometry):
        bounds = pygeos.bounds(bounds)
    elif isinstance(bounds, (list, tuple)):
        pass
    else:
        raise Exception("bounds type unknown!")
    
    rasterization_profile = extents_to_profile(bounds, gsd=gsd, crs=crs, count=3, dtype=rio.uint8)
    polygon_rasterized = rasterize_from_profile(gdf.geometry, rasterization_profile, 1)
    polygon_rasterized = binary_to_multiclass(polygon_rasterized)
    if blur:
        polygon_rasterized = np.transpose(gaussian_filter(np.transpose(polygon_rasterized, (1,2,0)).astype(float), sigma=blur), (2,0,1))
    return polygon_rasterized

def polygons_to_weighted_multiclass(gdf, bounds, crs, gsd=0.5, **kwargs ):
    """
    """
    weight_col_name = kwargs.pop("weight_col")
    class_labels = polygons_to_multiclass(gdf.geometry, bounds, crs, gsd,**kwargs)
    
    if isinstance(bounds, pygeos.Geometry):
        bounds = pygeos.bounds(bounds)
    elif isinstance(bounds, (list, tuple)):
        pass
    else:
        raise Exception("bounds type unknown!")
    
    rasterization_profile = extents_to_profile(bounds, gsd=gsd, crs=crs, count=3, dtype=rio.float32)
    if gdf.empty:
        weight_values = None
    else:
        weight_values = gdf[weight_col_name].values
    weights_rasterized = rasterize_from_profile(gdf.geometry, rasterization_profile, weight_values, fill=0)
    total_widths = kwargs.get("contours_width") + kwargs.get("inner_width") + 1
    class_weights = np.expand_dims(morphology.dilation(weights_rasterized, morphology.square(total_widths)), 0)
    class_weights[class_weights==0]=1
    return class_labels, class_weights
    