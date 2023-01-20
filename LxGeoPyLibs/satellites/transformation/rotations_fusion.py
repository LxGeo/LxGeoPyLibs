
from functools import partial
import rasterio as rio
from enum import Enum
import numpy as np
import click
from rasterio.windows import Window
from scipy.stats import circmean
from tqdm import tqdm


class aggregationMethod(Enum):
    max = 1
    min = 2
    median = 3
    mean = 4
    circmean = 5
    circmean2 = 6


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

def aggregate_arrays(stacked: np.ndarray, method: aggregationMethod, nodata=None, axis=0, **kwargs):
    """
    Aggregates input arrays using method along axis
    """
    #stacked = stacked.astype("float"); stacked[stacked==nodata]=np.nan
    if method == aggregationMethod.max:
        return reversed_bands_cummulation(stacked.max(axis))
    if method == aggregationMethod.min:
        return reversed_bands_cummulation(stacked.min(axis))
    if method == aggregationMethod.median:
        return np.nanmedian(stacked, axis)
    if method == aggregationMethod.mean:
        return np.nanmean(stacked,axis)
    if method == aggregationMethod.circmean:
        circmean_func = partial(circmean, high=kwargs.get("circmean_high",np.pi), low=kwargs.get("circmean_low",0), nan_policy="omit")
        return np.apply_along_axis(circmean_func, 0, stacked)
    if method == aggregationMethod.circmean2:
        high=kwargs.get("circmean_high",np.pi); low=kwargs.get("circmean_low",0)
        period_length = high-low
        stacked_rad = 2 * np.pi *(stacked-low) / period_length
        mean_cos = np.nanmean(np.cos(stacked_rad), axis=axis)
        mean_sin = np.nanmean(np.sin(stacked_rad), axis=axis)
        atag = np.arctan2(mean_sin,mean_cos) % (2*np.pi)
        unwraped = low + period_length * atag/(2*np.pi)
        return unwraped
    

def get_stacked_view(datasets_map, st_row, st_col, tile_size):
    """
    """
    rotated_views_list = []
    for k,v in datasets_map.items():
        loaded_array = v.read(window=Window(st_row, st_col, tile_size, tile_size)).astype('float')
        #loaded_array[loaded_array==v.profile["nodata"]] = np.nan
        rotated_views_list.append(loaded_array)
    return np.stack(rotated_views_list)



@click.command()
@click.option('--in_image', '-i', multiple=True)
@click.option('--out_raster_path', '-o', required=True, type=click.Path())
@click.option("-aggm", "--aggregation_method", type=click.Choice(aggregationMethod.__members__), 
              callback=lambda c, p, v: getattr(aggregationMethod, v) if v else None, default="max")
@click.option('--tile_size', '-ts', default=256)
@click.option('--circmean_high', '-cmh', default=256)
@click.option('--circmean_low', '-cml', default=0)
def rotation_fusion(in_image, out_raster_path, aggregation_method, tile_size, circmean_high, circmean_low):
    
    images_set = set(in_image)
    assert len(images_set) == len(in_image), "Duplicate image path provided!"

    rio_datasets_map = { k: rio.open(k) for k in images_set }
    sample_dst = list(rio_datasets_map.values())[0]
    out_profile=sample_dst.profile.copy()

    out_profile.update({"tiled": True, "blockxsize":tile_size,"blockysize":tile_size})
    
    with rio.open(out_raster_path, "w", **out_profile) as tar_dst:
        for c_row_start in tqdm(range(0,out_profile["height"], tile_size), desc="Progress!"):
            for c_col_start in range(0,out_profile["width"], tile_size):
                agg_result = aggregate_arrays(get_stacked_view(rio_datasets_map, c_row_start, c_col_start, tile_size ),
                 method=aggregation_method, nodata=out_profile["nodata"], circmean_high=circmean_high, circmean_low=circmean_low
                )
                tar_dst.write(agg_result, window = Window(c_row_start, c_col_start, agg_result.shape[-1], agg_result.shape[-2]))
    
if __name__ == "__main__":
    rotation_fusion()

