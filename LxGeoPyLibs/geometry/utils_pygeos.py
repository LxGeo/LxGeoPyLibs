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
        
        if not all_rings:
            return []
        polygons = pygeos.polygons(all_rings, indices=indices)
        return polygons

    
    def points_loader(coords):
        points = pygeos.points(coords)
        return points

    def linestring_loader(coords):
        
        all_linestrings=[]
        indices=[]
        for idx, c_linestring in enumerate(coords):
            all_linestrings.extend(c_linestring)
            indices.extend([idx]*len(c_linestring))
        
        if not all_linestrings:
            return []
        
        linestrings = pygeos.linestrings(all_linestrings, indices=indices)
        return linestrings

    if geom_type.lower() == "polygon":
        return polygon_loader
    elif geom_type.lower() == "points":
        return points_loader
    elif geom_type.lower() == "linestring":
        return linestring_loader
    else:
        raise Exception(f"Geometry type '{geom_type}' creator is not provided.")