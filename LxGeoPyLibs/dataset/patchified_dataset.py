import numpy as np
from LxGeoPyLibs.geometry.grid import make_grid
import pygeos
import math
import rasterio as rio
import torch
from LxGeoPyLibs.vision.image_transformation import Trans_Identity
import tqdm
from LxGeoPyLibs.geometry.utils_rio import extents_to_profile
from LxGeoPyLibs.ppattern.fixed_size_dict import FixSizeOrderedDict
from typing import Any

class PatchifiedDataset(object):
    """
    
    """

    def __init__(self, spatial_patch_size, spatial_patch_overlap, bounds_geom):
        super().__init__()

        """
        Setup patch loading settings using spatial coordinates.
        Args:
            patch_size: a couple of positive integers in spatial metric.
            patch_overlap: a positive integer in spatial metric.
            bounds_geom: pygeos polygon
        """

        self.spatial_patch_size = spatial_patch_size
        self.spatial_patch_overlap = spatial_patch_overlap
        self.bounds_geom = bounds_geom
        # buffer bounds_geom to include out of bound area
        buff_bounds_geom = pygeos.buffer(bounds_geom, self.spatial_patch_overlap, cap_style="square", join_style="mitre")

        grid_step = self.spatial_patch_size[0]-self.spatial_patch_overlap*2, self.spatial_patch_size[1]-self.spatial_patch_overlap*2
        assert grid_step[0]>0 and grid_step[1]>0 , "Spatial patch overlap is high! Reduce patch overlap."
        self.patch_grid = make_grid(buff_bounds_geom, grid_step[0], grid_step[1], self.spatial_patch_size[0], self.spatial_patch_size[1],
                                    filter_predicate = lambda x: pygeos.intersects(x, bounds_geom) )
        
        self.is_setup=True
    
    def __len__(self):
        return len(self.patch_grid)
    
    def __getitem__(self, index):
        return self.patch_grid[index]
    
    def gsd(self):
        raise NotImplementedError

    def get_stacked_batch(self, input_to_stack:[Any]):
        raise NotImplementedError

    def predict_to_file(self, out_file, model, tile_size=(256,256), post_processing_fn=lambda x:x, augmentations=None ):
        """
        Runs prediction and postprocessing if provided using prediction model and save to raster.
        Args:
            -out_file: string path to output file
            -model: callable prediction model having following methods (batch_size, min_patch_size )
            -tile_size: tile size is multiple of 16.
            -post_processing_fn: a callable to postprocess prediction. Must be callable on 4d tensors (batched).
            -augmentations: a list of augmentation must be from LxGeoPylibs.vision.imageTransformation or callable
             that takes two position parameters as images (image, gt) and returns a tuple of transformed images.        
        """

        if augmentations is None:
            augmentations=[Trans_Identity()]
        else:
            augmentations = augmentations
        
        batch_size = model.batch_size()
        min_patch_size = model.min_patch_size()
        assert tile_size[0]>min_patch_size, "Tile size is lower than minimum patch size of the model!"
        
        patch_size = (
            (math.floor(tile_size[0]/min_patch_size)+1)*min_patch_size,
            (math.floor(tile_size[1]/min_patch_size)+1)*min_patch_size
         )
        patch_overlap = (patch_size[0]-tile_size[0])//2
        # setup tile loading 
        self.setup_pixel(patch_size, patch_overlap)

        # temp post processing out type
        sample_input = self.get_stacked_batch([self[0]]*batch_size)
        # to device
        sample_input = [s.to(model.device) for s in sample_input]
        with torch.no_grad():
            sample_output = post_processing_fn(model.predict_step( sample_input ))
            if type(sample_output) == torch.Tensor:
                sample_output = sample_output.cpu().numpy()
        out_band_count=sample_output.shape[-3]

        out_profile = extents_to_profile(pygeos.bounds(self.bounds_geom), gsd = self.gsd())
        out_profile.update({"count": out_band_count, "dtype":sample_output.dtype, "tiled": True, "blockxsize":tile_size[0],"blockysize":tile_size[1], "crs":self.crs()})
        
        with rio.open(out_file, "w", **out_profile) as target_dst:
            
            target_bound_window = rio.windows.Window(0,0, target_dst.width, target_dst.height)

            with torch.no_grad():
                
                def combine_and_write_tile(item):
                    """
                    Function to combine prediction of a single augmented patch and crop extra pixels using overlap value and finally save to dataset
                    """
                    tile_idx, prediction_list = item
                    #mean_patch_pred = torch.stack(prediction_list).mean(dim=0)
                    mean_patch_pred = np.stack(prediction_list, axis=0).mean(axis=0)

                    c_patch_geom = self.patch_grid[tile_idx]
                    c_tile_geom = pygeos.buffer(c_patch_geom, -self.spatial_patch_overlap, cap_style="square", join_style="mitre")
                    c_tile_window = rio.windows.from_bounds(*pygeos.bounds(c_tile_geom), transform=out_profile["transform"])
                    
                    #cropping tile window within target dataset bounds
                    if not rio.windows.intersect(target_bound_window,c_tile_window):
                        return
                    c_cropped_tile_window =  target_bound_window.intersection(c_tile_window)
                    col_left_shift = c_cropped_tile_window.col_off - c_tile_window.col_off
                    row_up_shift = c_cropped_tile_window.row_off - c_tile_window.row_off

                    tile_pred = mean_patch_pred[
                        :,
                        math.floor(patch_overlap+row_up_shift):math.floor(patch_overlap+row_up_shift+c_cropped_tile_window.height),
                        math.floor(patch_overlap+col_left_shift):math.floor(patch_overlap+col_left_shift+c_cropped_tile_window.width)
                        ]
                    target_dst.write(tile_pred,window=c_cropped_tile_window)
                    return

                to_predict_queue = [] # a list of tuples (tile_index, tile_array)
                MAX_CACHE_SIZE=1+(batch_size//len(augmentations))
                tile_pred_cache=FixSizeOrderedDict(max=MAX_CACHE_SIZE, on_del_lambda=combine_and_write_tile,
                 before_delete_check=lambda x:len(x[1])==len(augmentations)
                 )

                def process_per_batch(to_predict_queue):
                    """
                    loads items from queue and runs prediction per batch and add to cache
                    """
                    c_batch = to_predict_queue[:batch_size]; del to_predict_queue[:batch_size]
                    # unzip c_batch
                    c_tiles_indices, c_batch = list(zip(*c_batch))
                    if len(c_batch)<batch_size:
                        missing_items_count = batch_size - len(c_batch)
                        c_batch = list(c_batch) + [c_batch[0]]*missing_items_count 
                    
                    c_batch = self.get_stacked_batch(c_batch)
                    c_batch = [s.to(model.device) for s in c_batch]
                    preds = model.predict_step(c_batch)
                    post_preds = post_processing_fn(preds)#.cpu()
                    if type(post_preds)==torch.Tensor and post_preds.is_cuda:
                        post_preds = post_preds.cpu().numpy()
                    for c_tile_idx, c_pred in zip(c_tiles_indices, post_preds[:]):
                        tile_pred_cache.setdefault(c_tile_idx, []).append(c_pred)
                    
                    return

                for c_tile_idx, c_tile in tqdm.tqdm(enumerate(self), total=len(self)):
                    # add augmented tile to queue
                    to_predict_queue.extend( [(c_tile_idx, aug(c_tile, None)[0]) for aug in augmentations] )

                    if len(to_predict_queue)>=batch_size:
                        process_per_batch(to_predict_queue)
                
                # finish last cached items
                if to_predict_queue: process_per_batch(to_predict_queue)
                for _ in range(len(tile_pred_cache)): tile_pred_cache.popitem()



class CallableModel():

    def __init__(self, bs=1, mps=128):
        super().__init__()
        self.bs=bs
        self.mps = mps

    def __call__(self, x):
        return self.forward(x)
    
    def batch_size(self):
        return self.bs
    
    def min_patch_size(self):
        return self.mps
    
    def predict_step(self, batch, batch_idx: int = None, dataloader_idx: int = 0):
        return self(batch)
