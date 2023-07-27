from functools import lru_cache, cached_property
import os
import math
import rasterio as rio
import rasterio.transform
import numpy as np
import torch
from torch.utils.data import Dataset
from LxGeoPyLibs.vision.image_transformation import Trans_Identity
import multiprocessing
from LxGeoPyLibs.geometry.grid import make_grid
from LxGeoPyLibs.dataset.common_interfaces import BoundedDataset
from LxGeoPyLibs.dataset.patchified_dataset import PatchifiedDataset, PixelPatchifiedDataset
import pygeos
import tqdm
from LxGeoPyLibs.geometry.utils_rio import extents_to_profile, window_round
from LxGeoPyLibs.ppattern.fixed_size_dict import FixSizeOrderedDict
from LxGeoPyLibs.ppattern.exceptions import MissingFileException
from LxGeoPyLibs import _logger

class RasterRegister(dict):

    def __init__(self):
        super(RasterRegister, self).__init__()
    
    def __del__(self):
        for k,v in self.items():
            print("Closing raster at {}".format(k))
            v.close()

rasters_map=RasterRegister()


class RasterDataset(PixelPatchifiedDataset):
    
    READ_RETRY_COUNT = 4
    DEFAULT_PATCH_SIZE = (256,256)
    DEFAULT_PATCH_OVERLAP = 100

    def __init__(self, image_path=None, augmentation_transforms=None,preprocessing=None, bounds_geom=None, pixel_patch_size=None, pixel_patch_overlap=None):
                        
        if not os.path.isfile(image_path):
            raise MissingFileException(image_path)
        
        self.image_path=image_path
        self.locker = multiprocessing.Lock()

        rasters_map.update({
            self.image_path: rio.open(self.image_path)
            })
        raster_total_bound_geom = pygeos.box(*self.bounds)
        if bounds_geom:
            assert pygeos.intersects(raster_total_bound_geom, bounds_geom), "Boundary geometry is out of raster extents!"
            bounds_geom = bounds_geom
        else:
            bounds_geom = raster_total_bound_geom
        
        BoundedDataset.__init__(self, bounds_geom, self.rio_profile["crs"])
        PixelPatchifiedDataset.__init__(self, self.rio_profile["transform"][0], -self.rio_profile["transform"][4])

        if augmentation_transforms is None:
            self.augmentation_transforms=[Trans_Identity()]
        else:
            self.augmentation_transforms = augmentation_transforms        
        
        self.Y_size, self.X_size = self.rio_profile["height"], self.rio_profile["width"] ## Check if correct (HW or WH)
        
        self.preprocessing=preprocessing
        self.is_setup=False
        if not None in (pixel_patch_size, pixel_patch_overlap):
            self.setup_patch_per_pixel(pixel_patch_size, pixel_patch_overlap, self.bounds_geom)
    
    @cached_property
    def rio_profile(self):
        with rio.open(self.image_path) as dst:
            profile = dst.profile
        return profile        
    
    def rio_dataset(self):
        if not self.image_path in rasters_map:
            rasters_map[self.image_path] = rio.open(self.image_path)
        return rasters_map[self.image_path]
    
    def setup_patch_per_spatial_unit(self, patch_size_spatial, patch_overlap_spatial, bounds_geom):
        """
        Setup patch loading settings using spatial coordinates.
        Args:
            patch_size_spatial: a tuple of positive integers in coords metric.
            patch_size_spatial: a positive integer in coords metric.
            bounds_geom: pygeos polygon
        """
        patch_size= (int(patch_size_spatial[0]/self.gsd()), int(patch_size_spatial[1]/self.gsd() ))
        patch_overlap= int(patch_overlap_spatial/self.gsd())
        self.setup_patch_per_pixel(patch_size, patch_overlap, bounds_geom)

    @cached_property
    def gsd(self):
        return abs(self.rio_profile["transform"][0])
    
    @cached_property
    def bounds(self):
        return rasterio.transform.array_bounds(self.rio_profile["height"], self.rio_profile["width"], self.rio_profile["transform"])

    def __len__(self):
        return PixelPatchifiedDataset.__len__(self)*len(self.augmentation_transforms)
    
    @lru_cache
    def _load_padded_raster_window(self, window_geom, patch_size=None):
        """
        Function to load image data by window and applying respective padding if requiered.
        """

        c_window = rio.windows.from_bounds(*pygeos.bounds(window_geom), transform=self.rio_profile["transform"]).round_offsets()
        
        img=None
        for _ in range(self.READ_RETRY_COUNT):
            with self.locker:
                try:
                    img = self.rio_dataset().read(window=c_window)
                    break
                except rio.errors.RasterioIOError as e:
                    _logger.warn(f"Error reading window from rasterio dataset of raster at {self.image_path}")
        
        if img is None:
            raise(Exception(f"Error loading image after {self.READ_RETRY_COUNT} trials!"))
        
        if not patch_size:
            patch_size = self.pixel_patch_size
        assert patch_size, "Patch size is not set for loading padded windows!"
        ## padding check
        left_pad = int(-min(0, c_window.col_off))
        right_pad = int(max(self.X_size, c_window.col_off+patch_size[0]) - self.X_size)
        up_pad = int(-min(0, c_window.row_off))
        down_pad = int(max(self.Y_size, c_window.row_off+patch_size[1]) - self.Y_size)
        if any([left_pad, right_pad, up_pad, down_pad]):
            pad_sett = (0,0),(up_pad, down_pad), (left_pad, right_pad)
            img = np.pad(img, pad_sett)
        
        return img
    
    def __getitem__(self, idx):
        
        assert self.is_setup, "Dataset is not set up!"
        window_idx = idx // (len(self.augmentation_transforms))
        transform_idx = idx % (len(self.augmentation_transforms))
        
        window_geom = PatchifiedDataset.__getitem__(self, window_idx)
        
        img = self._load_padded_raster_window(window_geom)
        
        c_trans = self.augmentation_transforms[transform_idx]
        img, _, _ = c_trans(img, img)
        
        if self.preprocessing:
            img = self.preprocessing(img)
        
        img = torch.from_numpy(img).float()
        
        return img
    
    def get_stacked_batch(self, input_to_stack):
        #return [torch.stack(input_to_stack)]
        return torch.stack(input_to_stack)
    
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
        self.setup_patch_per_pixel(patch_size, patch_overlap, self.bounds_geom)

        # temp post processing out type
        sample_input = self.get_stacked_batch([self[0]]*batch_size)
        # to device
        #sample_input = [s.to(model.device) for s in sample_input]
        with torch.no_grad():
            sample_output = model.predict_step( sample_input )
            sample_output = post_processing_fn(sample_output)
            if type(sample_output) == torch.Tensor:
                sample_output = sample_output.cpu().numpy()
        out_band_count=sample_output.shape[-3]

        out_profile = extents_to_profile(pygeos.bounds(self.bounds_geom), gsd = self.gsd())
        out_profile.update({"count": out_band_count, "dtype":sample_output.dtype, "tiled": True, "blockxsize":tile_size[0],"blockysize":tile_size[1], "crs":self.crs})
        
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
                    c_tile_window = window_round(c_tile_window)

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
                    
                    c_batch = self.get_stacked_batch([s for s in c_batch])
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


from LxGeoPyLibs.dataset.patchified_dataset import CallableModel
if __name__ == "__main__":

    in_file = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Pakistan_Rawalpindi_B_Neo/preds/build_probas.tif"
    c_r = RasterDataset(in_file)
    mdl = CallableModel(lambda x:np.expand_dims(np.argmax(x[0],axis=1),0).astype(np.uint8))
    out_file = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Pakistan_Rawalpindi_B_Neo/preds/build_labels.tif"
    from functools import partial
    bands_combiner = None#partial(torch.sum, dim=1, keepdim=True)
    c_r.predict_to_file(out_file, mdl)
    #c_r.predict_to_file(out_file, mdl, post_processing_fn=bands_combiner, tile_size=(256,256))
    pass