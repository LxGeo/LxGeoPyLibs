
def get_common_extents(*extents):
    """
    Return common extents of all arguments
    """
    assert len(extents)>0, "Input extents arguments missing!"
    
    minx, miny, maxx, maxy = extents[0]
    
    for ext in extents:
        minx = min(minx, ext[0])
        miny = min(miny, ext[1])
        maxx = min(maxx, ext[2])
        maxy = min(maxy, ext[3])
    
    return (minx, miny, maxx, maxy)