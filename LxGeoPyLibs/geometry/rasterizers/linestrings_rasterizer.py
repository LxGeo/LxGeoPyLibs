
import pygeos
from LxGeoPyLibs.geometry.utils_rio import extents_to_profile
from LxGeoPyLibs.geometry.rasterizers import rasterize_from_profile
from LxGeoPyLibs.ppattern.customized_arguments import customize_arg
from LxGeoPyLibs.ppattern.common_argument_extension import to_pygeos_bounds
import rasterio as rio
from skimage import morphology
import numpy as np

@customize_arg("bounds", to_pygeos_bounds)
def LineStringWA_to_displacment_map(gdf, bounds, crs, gsd=0.5, disp_x_column_name="disp_x",disp_y_column_name="disp_y", weight_column_name="hrel"):
    """
    """    
    flow_profile = extents_to_profile(bounds, crs=crs, count=3, dtype=np.float32, gsd=gsd)
    if gdf.empty:
        return np.zeros((flow_profile["count"], flow_profile["height"], flow_profile["width"]))

    dispx = rasterize_from_profile(gdf.geometry, flow_profile, gdf[disp_x_column_name].values / gsd)
    dispy = rasterize_from_profile(gdf.geometry, flow_profile, gdf[disp_y_column_name].values / -gsd)
    weight = rasterize_from_profile(gdf.geometry, flow_profile, gdf[weight_column_name].values, background_fill=1)
    weighted_flow = np.stack([dispx, dispy, weight])
    return weighted_flow


def linestring_to_multiclass(gdf, bounds, crs, contours_width=3, gsd=0.5):
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
        linestrings_rasterized = rasterize_from_profile(gdf.geometry, rasterization_profile, 1)
        linestrings_with_contours = morphology.dilation(linestrings_rasterized, morphology.square(contours_width))
        linestrings_contours = np.maximum(linestrings_with_contours.astype(int)-linestrings_rasterized.astype(int), 0)
        background_map = np.maximum( np.ones_like(linestrings_with_contours).astype(int)-linestrings_with_contours.astype(int),0 )

        proba_map = np.stack([background_map, linestrings_rasterized, linestrings_contours])
    return proba_map