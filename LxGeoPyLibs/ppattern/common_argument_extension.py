from LxGeoPyLibs.ppattern.customized_arguments import customize_arg


def to_pygeos_bounds(arg_val):
    """
    Argument agnostic function to create pygeos boundary object
    """
    import pygeos, shapely
    if isinstance(arg_val, pygeos.Geometry):
        return pygeos.bounds(arg_val)
    if isinstance(arg_val, shapely.Geometry ):
        return pygeos.box(*arg_val.bounds)
    elif isinstance(arg_val, (list, tuple)):
        assert len(arg_val)==4, "Cannot convert list of bounds of size different than 4 to bounds object"
        return arg_val
    elif isinstance(arg_val, str ):
        return pygeos.bounds(pygeos.from_wkt(arg_val))
    else:
        raise Exception(f"Argument type {type(arg_val)} is not included!")