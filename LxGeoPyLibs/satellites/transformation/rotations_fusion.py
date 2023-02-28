
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



class rotated_raster_manager():
    rotated_raster_keys = ["rot0", "rot90", "rot180", "rot270"]
    num_reverse_orientation_map = { "rot0":0, "rot90": 3, "rot180":2, "rot270":1 }
    DEFAULT_TILE_SIZE = 256
    def __init__(self, **kwargs):
        
        self.tile_size = kwargs.get("tile_size", self.DEFAULT_TILE_SIZE)
        self.datasets_dict = dict()
        #Open datasets
        for rk in self.rotated_raster_keys:
            raster_path = kwargs.get(rk)
            if raster_path:
                self.datasets_dict[rk] = rio.open(raster_path)
        assert len(rk)>0, "Provide at least one rotated image"        
    
    def get_main_orientation_profile(self):
        # Set output profile
        sample_key, sample_value = next(iter(self.datasets_dict.items()))
        if not  hasattr(self, "main_orientation_profile"):
            self.main_orientation_profile = self.init_profile(sample_key, sample_value.profile)
        return self.main_orientation_profile
    
    def height(self):
        return self.get_main_orientation_profile()["height"]
    
    def width(self):
        return self.get_main_orientation_profile()["width"]
    
    def init_profile(self, rotation_key, profile):
        out_profile = profile.copy()
        out_profile.update({"tiled": True, "blockxsize":self.tile_size,"blockysize":self.tile_size})
        if (rotation_key == "rot90") | (rotation_key == "rot270"):
            out_profile["height"], out_profile["width"] = out_profile["width"],out_profile["height"]
        return out_profile

    def get_respective_arrays(self, col_start, row_start, col_size, row_size):
        """
        Returns aligned arrays using main orientation col,row ranges
        """
        assert col_start+col_size <= self.get_main_orientation_profile()["width"], "Columns maximum exceeded"
        assert row_start+row_size <= self.get_main_orientation_profile()["height"], "Rows maximum exceeded"

        def transform_rect_range(rot_key, col_start, row_start, col_size, row_size):
            if rot_key == "rot0":
                return col_start, row_start, col_size, row_size
            elif rot_key == "rot90":
                return row_start, self.width() - col_start - col_size, row_size, col_size
            elif rot_key == "rot180":
                return self.width() - col_start -col_size, self.height() - row_start - row_size, col_size, row_size
            elif rot_key == "rot270":
                return self.height() - row_start - row_size, col_start, row_size, col_size

        def get_main_oriented_array(rot_key, st_col, st_row, col_size, row_size):
            st_col, st_row, col_size, row_size = transform_rect_range(rot_key, st_col, st_row, col_size, row_size)
            loaded_array = self.datasets_dict[rot_key].read(window=Window(st_col, st_row, row_size, col_size)).astype('float')
            reverse_rotate_cnt = self.num_reverse_orientation_map[rot_key]
            return np.rot90(loaded_array, reverse_rotate_cnt, axes=(1,2))

        all_arrays = [get_main_oriented_array(c_rot_key, col_start, row_start, col_size, row_size) for c_rot_key in self.datasets_dict.keys()]
        return np.stack(all_arrays)




@click.command()
@click.option('--in_image_param', '-i', multiple=True, nargs=2)
@click.option('--out_raster_path', '-o', required=True, type=click.Path())
@click.option("-aggm", "--aggregation_method", type=click.Choice(aggregationMethod.__members__), 
              callback=lambda c, p, v: getattr(aggregationMethod, v) if v else None, default="max")
@click.option('--tile_size', '-ts', default=256)
@click.option('--circmean_high', '-cmh', default=256)
@click.option('--circmean_low', '-cml', default=0)
def rotation_fusion_updated(in_image_param, out_raster_path, aggregation_method, tile_size, circmean_high, circmean_low):
    """
    Used in case input images are not rotated back to initial orientaton
    """
    in_image_paths, rotation_values = list(zip(*in_image_param))
    images_set = set(in_image_paths)
    assert len(images_set) == len(in_image_paths), "Duplicate image path provided!"

    mngr = rotated_raster_manager(**{v:k for k,v in in_image_param})
    out_profile = mngr.get_main_orientation_profile()
    with rio.open(out_raster_path, "w", **out_profile) as tar_dst:
        for c_row_start in tqdm(range(0,out_profile["height"], tile_size), desc="Progress!"):
            for c_col_start in range(0,out_profile["width"], tile_size):
                agg_result = aggregate_arrays(mngr.get_respective_arrays(c_col_start, c_row_start, tile_size, tile_size),
                 method=aggregation_method, nodata=out_profile["nodata"], circmean_high=circmean_high, circmean_low=circmean_low
                )
                tar_dst.write(agg_result, window = Window(c_col_start, c_row_start, agg_result.shape[-1], agg_result.shape[-2]))
    


if __name__ == "__main__":
    rotation_fusion_updated()

