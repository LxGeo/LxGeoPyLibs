
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
            band_nums = list(range(dst.profile["count"]+1))
    
    output_code = gdal_pansharpen(
    pan_name=input_pan_path,    
    spectral_names=[input_ms_path], 
    band_nums=[1, 2, 3],
    dst_filename=out_path)

    return output_code
