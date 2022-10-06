import inspect
from osgeo_utils.gdal_pansharpen import gdal_pansharpen
import rasterio as rio


def pansharpen_image(input_pan_path, input_ms_path, out_path, band_nums=None):
    """
    function used to pansharpen images.
    Args: self-explanatory
        band_nums: specify bands to pansharpen if None apply on all MS bands.
    Return: output exectution code (0 ran succefully) and (1 error)
    """

    if band_nums is None:
        with rio.open(input_ms_path) as dst:
            band_nums = list(range(1, dst.profile["count"]+1))
    
    
    
    if "pan_name" in inspect.getargspec(gdal_pansharpen).args:
        output_code = gdal_pansharpen(
        pan_name=input_pan_path,    
        spectral_names=[input_ms_path], 
        band_nums=band_nums,
        dst_filename=out_path)
    else:
        command_args = [""] + [str(item) for sublist in zip(["-b"]*len(band_nums), band_nums) for item in sublist] + [input_pan_path, input_ms_path, out_path ]
        output_code = gdal_pansharpen(command_args)

    return output_code
