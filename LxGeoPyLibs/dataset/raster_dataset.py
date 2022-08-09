from functools import lru_cache
import os
import math
import rasterio as rio
import numpy as np
import torch
from torch.utils.data import Dataset
from LxGeoPyLibs.vision.image_transformation import Trans_Identity
import multiprocessing
from LxGeoPyLibs.ppattern.fixed_size_dict import FixSizeOrderedDict
from LxGeoPyLibs.geometry.grid import make_grid
from shapely.geometry import box
import pygeos
import tqdm

class RasterRegister(dict):

    def __init__(self):
        super(RasterRegister, self).__init__()
    
    def __del__(self):
        for k,v in self.items():
            print("Closing raster at {}".format(k))
            v.close()

rasters_map=RasterRegister()
lock = multiprocessing.Lock()

class RasterDataset(Dataset):
    
    READ_RETRY_COUNT = 4
    DEFAULT_PATCH_SIZE = (256,256)
    DEFAULT_PATCH_OVERLAP = 100

    def __init__(self, image_path=None, augmentation_transforms=None,preprocessing=None, patch_size=None, patch_overlap=None):
                        
        assert os.path.isfile(image_path), f"Can't find raster in {image_path}"
        
        self.image_path=image_path

        if augmentation_transforms is None:
            self.augmentation_transforms=[Trans_Identity()]
        else:
            self.augmentation_transforms = augmentation_transforms
        
        rasters_map.update({
            self.image_path: rio.open(self.image_path)
            })
        
        self.Y_size, self.X_size = rasters_map[self.image_path].shape
             
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

        self.preprocessing=preprocessing
        self.is_setup=False
        if not None in (patch_size, patch_overlap):
            self.setup_spatial(patch_size, patch_overlap)
    
    def setup_spatial(self, patch_size, patch_overlap, bounds_geom=None):
        """
        Setup patch loading settings using spatial coordinates.
        Args:
            patch_size: a tuple of positive integers in pixels.
            patch_overlap: a positive integer in pixels.
            bounds_geom: pygeos polygon
        """
        self.patch_size= patch_size
        self.patch_overlap= patch_overlap
        
        # If no bounds provided, use image bounds
        if not bounds_geom:
            bounds_geom = pygeos.box(*rasters_map[self.image_path].bounds)

        pixel_x_size = rasters_map[self.image_path].transform[0]
        pixel_y_size = -rasters_map[self.image_path].transform[4]

        self.patch_size_spatial = (self.patch_size[0]*pixel_x_size, self.patch_size[1]*pixel_y_size)
        self.patch_overlap_spatial = self.patch_overlap*pixel_x_size

        # buffer bounds_geom to include out of bound area
        buff_bounds_geom = pygeos.buffer(bounds_geom, self.patch_overlap_spatial, cap_style="square", join_style="mitre")
        grid_step = self.patch_size_spatial[0]-self.patch_overlap_spatial*2, self.patch_size_spatial[1]-self.patch_overlap_spatial*2
        self.patch_grid = make_grid(buff_bounds_geom, grid_step[0], grid_step[1], self.patch_size_spatial[0], self.patch_size_spatial[1],
        filter_predicate = lambda x: pygeos.intersects(x, bounds_geom) )
        
        #tile_grid = pygeos.buffer(patch_grid, -2*self.patch_overlap, cap_style="square", join_style="mitre")
        self.is_setup=True

    def __len__(self):
        assert self.is_setup, "Dataset is not set up!"
        return len(self.patch_grid)*len(self.augmentation_transforms)
    
    @lru_cache
    def _load_padded_raster_window(self, window_geom):
        """
        Function to load image data by window and applying respective padding if requiered.
        """

        c_window = rio.windows.from_bounds(*pygeos.bounds(window_geom), transform=rasters_map[self.image_path].transform)
        
        lock.acquire()
        for _ in range(self.READ_RETRY_COUNT):
            try:
                img = rasters_map[self.image_path].read(window=c_window)
                break
            except rio.errors.RasterioIOError as e:
                lock.release()
        lock.release()

        ## padding check
        left_pad = int(-min(0, c_window.col_off))
        right_pad = int(max(self.X_size, c_window.col_off+self.patch_size[0]) - self.X_size)
        up_pad = int(-min(0, c_window.row_off))
        down_pad = int(max(self.Y_size, c_window.row_off+self.patch_size[1]) - self.Y_size)
        if any([left_pad, right_pad, up_pad, down_pad]):
            pad_sett = (0,0),(up_pad, down_pad), (left_pad, right_pad)
            img = np.pad(img, pad_sett)
        
        return img
    
    def __getitem__(self, idx):
        
        assert self.is_setup, "Dataset is not set up!"
        window_idx = idx // (len(self.augmentation_transforms))
        transform_idx = idx % (len(self.augmentation_transforms))
        
        window_geom = self.patch_grid[window_idx]
        
        img = self._load_padded_raster_window(window_geom)
        
        c_trans = self.augmentation_transforms[transform_idx]
        img, _ = c_trans(img, img)
        
        if self.preprocessing:
            img = self.preprocessing(img)
        
        img = torch.from_numpy(img).float()
        
        return img

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
        overlap = (patch_size[0]-tile_size[0])//2
        # setup tile loading 
        self.setup_spatial(patch_size, overlap)

        # temp post processing out type
        sample_output = post_processing_fn(model( torch.stack([self[0]]*batch_size) )).numpy()
        out_band_count=sample_output.shape[-3]

        out_profile = rasters_map[self.image_path].profile.copy()
        out_profile.update({"count": out_band_count, "dtype":sample_output.dtype, "tiled": True, "blockxsize":tile_size[0],"blockysize":tile_size[1]})
        
        with rio.open(out_file, "w", **out_profile) as target_dst:
            
            target_bound_geom = pygeos.box(*target_dst.bounds)

            with torch.no_grad():
                
                def combine_and_write_tile(item):
                    """
                    Function to combine prediction of a single augmented patch and crop extra pixels using overlap value and finally save to dataset
                    """
                    tile_idx, prediction_list = item
                    mean_patch_pred = torch.stack(prediction_list).mean(dim=0)

                    c_patch_geom = self.patch_grid[tile_idx]
                    c_tile_geom = pygeos.buffer(c_patch_geom, -self.patch_overlap_spatial, cap_style="square", join_style="mitre")
                    #cropping tile within target dataset
                    c_tile_geom = pygeos.intersection(c_tile_geom, target_bound_geom)
                    c_tile_window = rio.windows.from_bounds(*pygeos.bounds(c_tile_geom), transform=rasters_map[self.image_path].transform)

                    tile_pred = mean_patch_pred[
                        :,self.patch_overlap:self.patch_overlap+int(c_tile_window.width),
                        self.patch_overlap:self.patch_overlap+int(c_tile_window.height)]
                    target_dst.write(tile_pred,window=c_tile_window)
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
                        c_batch = list(c_batch) + [torch.zeros_like(c_batch[0])]*missing_items_count 
                    
                    c_batch = torch.stack(c_batch, 0)
                    preds = model(c_batch); post_preds = post_processing_fn(preds)
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



class test_model():

    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return x
    
    def batch_size(self):
        return 13
    
    def min_patch_size(self):
        return 128

if __name__ == "__main__":

    in_file = "../DATA_SANDBOX/lxFlowAlign/data/train_data/paris_ortho1_rooftop/flow.tif"
    c_r = RasterDataset(in_file)
    mdl = test_model()
    out_file = "../DATA_SANDBOX/out_file.tif"
    from functools import partial
    bands_combiner = partial(torch.sum, dim=1, keepdim=True)
    c_r.predict_to_file(out_file, mdl, post_processing_fn=bands_combiner, tile_size=(256,256))
    pass