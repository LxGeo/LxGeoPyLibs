from functools import lru_cache
import os
import math
import rasterio as rio
import numpy as np
import torch
from torch.utils.data import Dataset
from LxGeoPyLibs.vision.image_transformation import Trans_Identity
import multiprocessing
from LxGeoPyLibs.geometry.grid import make_grid
from LxGeoPyLibs.dataset.patchified_dataset import PatchifiedDataset
import pygeos
import tqdm
from LxGeoPyLibs.geometry.utils_rio import extents_to_profile

class RasterRegister(dict):

    def __init__(self):
        super(RasterRegister, self).__init__()
    
    def __del__(self):
        for k,v in self.items():
            print("Closing raster at {}".format(k))
            v.close()

rasters_map=RasterRegister()
lock = multiprocessing.Lock()

class RasterDataset(Dataset, PatchifiedDataset):
    
    READ_RETRY_COUNT = 4
    DEFAULT_PATCH_SIZE = (256,256)
    DEFAULT_PATCH_OVERLAP = 100

    def __init__(self, image_path=None, augmentation_transforms=None,preprocessing=None, bounds_geom=None, patch_size=None, patch_overlap=None):
                        
        assert os.path.isfile(image_path), f"Can't find raster in {image_path}"
        
        self.image_path=image_path

        if augmentation_transforms is None:
            self.augmentation_transforms=[Trans_Identity()]
        else:
            self.augmentation_transforms = augmentation_transforms
        
        rasters_map.update({
            self.image_path: rio.open(self.image_path)
            })
        
        self.Y_size, self.X_size = self.rio_dataset().shape
             
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        
        raster_total_bound_geom = pygeos.box(*self.rio_dataset().bounds)
        if bounds_geom:
            assert pygeos.intersects(raster_total_bound_geom, bounds_geom), "Boundary geometry is out of raster extents!"
            self.bounds_geom = bounds_geom
        else:
            self.bounds_geom = raster_total_bound_geom

        self.preprocessing=preprocessing
        self.is_setup=False
        if not None in (patch_size, patch_overlap):
            self.setup_spatial(patch_size, patch_overlap, self.bounds_geom)
    
    def rio_dataset(self):
        return rasters_map[self.image_path]
    
    ### should be fixed to meters not pixels
    def setup_pixel(self, patch_size, patch_overlap, bounds_geom=None):
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
            bounds_geom = self.bounds_geom

        pixel_x_size = self.rio_dataset().transform[0]
        pixel_y_size = -self.rio_dataset().transform[4]

        patch_size_spatial = (self.patch_size[0]*pixel_x_size, self.patch_size[1]*pixel_y_size)
        patch_overlap_spatial = self.patch_overlap*pixel_x_size

        super(Dataset, self).__init__(patch_size_spatial, patch_overlap_spatial, bounds_geom)
    
    def setup_spatial(self, patch_size_spatial, patch_overlap_spatial, bounds_geom=None):
        """
        Setup patch loading settings using spatial coordinates.
        Args:
            patch_size: a tuple of positive integers in coords metric.
            patch_overlap: a positive integer in coords metric.
            bounds_geom: pygeos polygon
        """
        if not bounds_geom:
            bounds_geom = self.bounds_geom
        self.patch_size= (int(patch_size_spatial[0]/self.gsd()), int(patch_size_spatial[1]/self.gsd() ))
        self.patch_overlap= int(patch_overlap_spatial/self.gsd())
        super().__init__(patch_size_spatial, patch_overlap_spatial, bounds_geom)

    def gsd(self):
        return abs(self.rio_dataset().transform[0])
    
    def crs(self):
        return self.rio_dataset().crs
    
    def bounds(self):
        return self.rio_dataset().bounds

    def __len__(self):
        assert self.is_setup, "Dataset is not set up!"
        return super(Dataset, self).__len__()*len(self.augmentation_transforms)
    
    @lru_cache
    def _load_padded_raster_window(self, window_geom, patch_size=None):
        """
        Function to load image data by window and applying respective padding if requiered.
        """

        c_window = rio.windows.from_bounds(*pygeos.bounds(window_geom), transform=self.rio_dataset().transform).round_offsets()
        
        lock.acquire()
        for _ in range(self.READ_RETRY_COUNT):
            try:
                img = self.rio_dataset().read(window=c_window)
                break
            except rio.errors.RasterioIOError as e:
                lock.release()
        lock.release()
        
        if not patch_size:
            patch_size = self.patch_size
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
        
        window_geom = super(Dataset, self).__getitem__(window_idx)
        
        img = self._load_padded_raster_window(window_geom)
        
        c_trans = self.augmentation_transforms[transform_idx]
        img, _, _ = c_trans(img, img)
        
        if self.preprocessing:
            img = self.preprocessing(img)
        
        img = torch.from_numpy(img).float()
        
        return img
    
    def get_stacked_batch(self, input_to_stack):
        return [torch.stack(input_to_stack)]

    
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