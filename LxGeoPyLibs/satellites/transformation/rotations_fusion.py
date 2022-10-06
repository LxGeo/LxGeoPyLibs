from inspect import stack
from math import remainder
import rasterio as rio
from enum import Enum
import numpy as np
import click
from rasterio.windows import Window
from scipy.special import softmax


class aggregationMethod(Enum):
    max = 1
    min = 2
    median = 3
    mean = 4


def reversed_bands_cummulation(array, max_val=255):
    """
    Kind of normalization with priority to revered order bands
    """
    rem = max_val * np.ones_like(array[0])
    for c_band_idx in range(array.shape[0]-1,0,-1):
        array[c_band_idx] = np.minimum(array[c_band_idx], rem)
        rem -= array[c_band_idx]    
    array[0]=rem
    return array

def aggregate_arrays(method: aggregationMethod, stacked: np.ndarray, axis=0):
    """
    Aggregates input arrays using method along axis
    """
    
    if method == aggregationMethod.max:
        return reversed_bands_cummulation(stacked.max(axis))
    if method == aggregationMethod.min:
        return reversed_bands_cummulation(stacked.min(axis))
    if method == aggregationMethod.median:
        return np.median(stacked, axis)
    if method == aggregationMethod.mean:
        return stacked.mean(axis)

def get_stacked_view(datasets_map, st_row, st_col, tile_size):
    """
    """
    rotated_views_list = []
    for k,v in datasets_map.items():
        rotated_views_list.append(v.read(window=Window(st_row, st_col, tile_size, tile_size)))
    return np.stack(rotated_views_list)



@click.command()
@click.option('--in_image', '-i', multiple=True)
@click.option('--out_raster_path', '-o', required=True, type=click.Path())
@click.option("-aggm", "--aggregation_method", type=click.Choice(aggregationMethod.__members__), 
              callback=lambda c, p, v: getattr(aggregationMethod, v) if v else None, default="max")
@click.option('--tile_size', '-ts', default=256)
def rotation_fusion(in_image, out_raster_path, aggregation_method, tile_size):
    
    images_set = set(in_image)
    assert len(images_set) == len(in_image), "Duplicate image path provided!"

    rio_datasets_map = { k: rio.open(k) for k in images_set }
    sample_dst = list(rio_datasets_map.values())[0]
    out_profile=sample_dst.profile.copy()

    out_profile.update({"tiled": True, "blockxsize":tile_size,"blockysize":tile_size})
    
    with rio.open(out_raster_path, "w", **out_profile) as tar_dst:
        for c_row_start in range(0,out_profile["height"], tile_size):
            for c_col_start in range(0,out_profile["width"], tile_size):
                agg_result = aggregate_arrays(aggregation_method, get_stacked_view(rio_datasets_map, c_row_start, c_col_start, tile_size ))
                tar_dst.write(agg_result, window = Window(c_row_start, c_col_start, agg_result.shape[-1], agg_result.shape[-2]))
    
if __name__ == "__main__":
    rotation_fusion()

