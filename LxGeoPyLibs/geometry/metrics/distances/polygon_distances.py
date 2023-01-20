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
def prepare_dist_func(dist_func, distance_name):

    class distance_caller:
        def __init__(self, callable, name):
            self.name=name
            self.callable=callable
        def __call__(self, *args, **kwds):
            return self.callable(*args, *kwds)

    return distance_caller(np.vectorize(partial(multipolygon_option_wrapper(dist_func), symmetrise="average")), distance_name)

chamfer_distance = prepare_dist_func(chamfer_distance, "chamfer_distance")
hausdorff_distance = prepare_dist_func(hausdorff_distance, "hausdorff_distance")
polis_distance = prepare_dist_func(polis_distance, "polis_distance")

