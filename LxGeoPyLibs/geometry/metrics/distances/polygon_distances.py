from x2polygons.polygon_distance import chamfer_distance, hausdorff_distance, polis_distance
from functools import partial
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import itertools

# Adding multipolygon support
def multipolygon_option_wrapper(dist_func):

    def wrapped_func(geom1, geom2, **kwargs):
        
        if type(geom1) == MultiPolygon:
            geoms1 = list(geom1)
        else:
            geoms1 = [geom1]

        if type(geom2) == MultiPolygon:
            geoms2 = list(geom2)
        else:
            geoms2 = [geom2]
        
        return np.average([dist_func(g1,g2, **kwargs) for g1,g2 in list(itertools.product(geoms1, geoms2))])
    
    return wrapped_func

# preparing distance functions
def prepare_dist_func(dist_func):
    return np.vectorize(partial(multipolygon_option_wrapper(dist_func), symmetrise="average"))

chamfer_distance = prepare_dist_func(chamfer_distance)
hausdorff_distance = prepare_dist_func(hausdorff_distance)
polis_distance = prepare_dist_func(polis_distance)
