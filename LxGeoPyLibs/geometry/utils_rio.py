import rasterio as rio

def extents_to_profile(extent, gsd=0.5, **kwargs):
    
    in_width = round((extent[2]-extent[0])/gsd)
    in_height = round((extent[3]-extent[1])/gsd)
    in_transform = rio.transform.from_origin(extent[0], extent[-1], gsd,gsd)
    rasterization_profile = {
    "driver": "GTiff", "count":1, "height": in_height, "width":in_width,
    "dtype":rio.uint8, "transform":in_transform
    }
    rasterization_profile.update(kwargs)
    return rasterization_profile
