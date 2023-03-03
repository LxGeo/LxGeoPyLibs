import rasterio as rio
from rasterio import windows

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

def window_round(in_window):
    return rio.windows.Window(
        round(in_window.col_off), round(in_window.row_off),
        round(in_window.width), round(in_window.height)
    )

def get_required_padding(window, width, height):
    """
    int left_pad = -std::min<int>(xStart, 0);
    int right_pad = std::max<int>(xStart + xSize, image.cols) - image.cols;
    int top_pad = -std::min<int>(yStart, 0);
    int down_pad = std::max<int>(yStart + ySize, image.rows) - image.rows;
    """
    left_pad = -min(window.col_off,0)
    right_pad = max(window.col_off+window.width, width) - width
    top_pad = -min(0, window.row_off)
    bottom_pad = max(window.row_off + window.height, height) - height
    return ( (top_pad, bottom_pad), (left_pad, right_pad) )
