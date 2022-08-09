import pygeos
import numpy as np
from functools import partial

def get_pygeos_transformer(transformer):

    def transformation(coords):
        x, y = transformer.transform(*coords.T)
        return np.array([x, y]).T
    shape_apply = partial(pygeos.apply, transformation=transformation)

    return shape_apply

def get_pygeos_geom_creator(geom_type):
    """
    Returns a callable to load geometreies coordinates given geometry type
    """

    def polygon_loader(coords):
        
        all_rings=[]
        indices=[]
        for idx, c_poly_rings in enumerate(coords):
            for c_ring in c_poly_rings:
                all_rings.append(pygeos.linearrings(c_ring))
                indices.append(idx)
        
        polygons = pygeos.polygons(all_rings, indices=indices)
        return polygons

    
    def points_loader(coords):
        points = pygeos.points(coords)
        return points

    def linestring_loader(coords):
        linestrings = pygeos.linestrings(coords)
        return linestrings

    if geom_type.lower() == "polygon":
        return polygon_loader
    elif geom_type.lower() == "points":
        return points_loader
    elif geom_type.lower() == "linestring":
        return linestring_loader
    else:
        raise Exception(f"Geometry type '{geom_type}' creator is not provided.")