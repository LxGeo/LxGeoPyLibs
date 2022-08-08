import os
import math
from typing import DefaultDict
import rasterio as rio
import numpy as np
import torch
from torch.utils.data import Dataset
from LxGeoPyLibs.vision.image_transformation import Trans_Identity
import multiprocessing 
from collections import defaultdict
from LxGeoPyLibs.ppattern.fixed_size_dict import FixSizeOrderedDict
import tqdm
from shapely.geometry import box
import geopandas as gpd

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

    def __init__(self, image_path=None, augmentation_transforms=None,preprocessing=None, patch_size=(256,256), patch_overlap=(100,100)):
                        
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
        self.setup()
                
    
    def setup(self, patch_size=None, patch_overlap=None):
        """
        Setup patch loading settings
        """
        if patch_size: self.patch_size= patch_size
        if patch_overlap: self.patch_overlap= patch_overlap

        self.window_x_starts = np.arange(-self.patch_overlap[0], self.X_size-self.patch_overlap[0], self.patch_size[0]-self.patch_overlap[0]*2)
        #if (self.X_size-1)%self.patch_size[0] != 0: self.window_x_starts.append(self.X_size-1-self.patch_size[0])
        self.window_y_starts = np.arange(-self.patch_overlap[1], self.Y_size-self.patch_overlap[1], self.patch_size[1]-self.patch_overlap[1]*2)
        #if (self.Y_size-1)%self.patch_size[1] != 0: self.window_y_starts.append(self.Y_size-1-self.patch_size[1])
        self.is_setup=True

    def __len__(self):
        assert self.is_setup, "Dataset is not set up!"
        x_count = len(self.window_x_starts)
        y_count = len(self.window_y_starts)
        return x_count*y_count*len(self.augmentation_transforms)
    
    def __getitem__(self, idx):
        
        assert self.is_setup, "Dataset is not set up!"
        window_idx = idx // (len(self.augmentation_transforms))
        transform_idx = idx % (len(self.augmentation_transforms))
        
        window_x_start = self.window_x_starts[window_idx//len(self.window_y_starts)]
        window_y_start = self.window_y_starts[window_idx%len(self.window_y_starts)]
        c_window = rio.windows.Window(window_x_start, window_y_start, *self.patch_size )
        
        lock.acquire()
        for _ in range(self.READ_RETRY_COUNT):
            try:
                img = rasters_map[self.image_path].read(window=c_window)
                break
            except rio.errors.RasterioIOError as e:
                lock.release()
        lock.release()

        ## padding check
        left_pad = -min(0, window_x_start)
        right_pad = max(self.X_size, window_x_start+self.patch_size[0]) - self.X_size
        up_pad = -min(0, window_y_start)
        down_pad = max(self.Y_size, window_y_start+self.patch_size[1]) - self.Y_size
        if any([left_pad, right_pad, up_pad, down_pad]):
            pad_sett = (0,0),(up_pad, down_pad), (left_pad, right_pad)
            img = np.pad(img, pad_sett, mode="reflect")
        
        c_trans = self.augmentation_transforms[transform_idx]
        img, _ = c_trans(img, img)
        
        if self.preprocessing:
            img = self.preprocessing(img)
        
        img = torch.from_numpy(img).float()
        
        return img

    def predict_to_file(self, out_file, model, tile_size=(256,256), post_processing_fn=lambda x:x, augmentations=None ):
        """
        
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
        overlap = (
            (patch_size[0]-tile_size[0])//2,
            (patch_size[1]-tile_size[1])//2
        )
        # setup tile loading 
        self.setup(patch_size, overlap)

        # temp post processing out type
        sample_output = post_processing_fn(model( torch.stack([self[0]*batch_size]) )).numpy()
        out_band_count=sample_output.shape[1]

        out_profile = rasters_map[self.image_path].profile.copy()
        out_profile.update({"count": out_band_count, "dtype":sample_output.dtype, "tiled": True, "blockxsize":tile_size[0],"blockysize":tile_size[1]})
        
        with rio.open(out_file, "w", **out_profile) as target_dst:

            gdf = gpd.GeoDataFrame([(None,None,box(*target_dst.bounds))], columns=["idx","type", "geometry"], crs=target_dst.crs)
            gdf.to_file("../DATA_SANDBOX/out_file.shp")
            
            #model.model.training=False
            with torch.no_grad():
                
                def combine_and_write_tile(item):
                    """
                    Function to combine prediction of a single augmented patch and crop extra pixels using overlap value and finally save to dataset
                    """
                    tile_idx, prediction_list = item
                    mean_patch_pred = torch.stack(prediction_list).mean(dim=0)
                    c_tile_x_start = self.window_x_starts[tile_idx//len(self.window_y_starts)] + overlap[0]
                    c_tile_y_start = self.window_y_starts[tile_idx%len(self.window_y_starts)] + overlap[1]
                    tile_x_size = min(tile_size[0], self.X_size-c_tile_x_start)
                    tile_y_size = min(tile_size[1], self.Y_size-c_tile_y_start)
                    tile_pred = mean_patch_pred[:,overlap[0]:overlap[0]+tile_y_size, overlap[1]:overlap[1]+tile_x_size]
                    c_window = rio.windows.Window(c_tile_x_start, c_tile_y_start, tile_x_size, tile_y_size )
                    target_dst.write(tile_pred,window=c_window)

                    # temp for window check
                    c_tile_window = rio.windows.Window(c_tile_x_start, c_tile_y_start, tile_size[0], tile_size[1] )
                    tile_geom = box(*rio.windows.bounds(c_tile_window,target_dst.transform))
                    c_patch_window = rio.windows.Window(self.window_x_starts[tile_idx//len(self.window_y_starts)],
                     self.window_y_starts[tile_idx%len(self.window_y_starts)], self.patch_size[0], self.patch_size[1] )
                    patch_geom = box(*rio.windows.bounds(c_patch_window,target_dst.transform))
                    window_geom_tuples=[]
                    window_geom_tuples.append( (tile_idx, 0, tile_geom) )
                    window_geom_tuples.append( (tile_idx, 1, patch_geom) )
                    gdf = gpd.GeoDataFrame(window_geom_tuples, columns=["idx","type", "geometry"], crs=target_dst.crs)
                    gdf.to_file("../DATA_SANDBOX/out_file.shp", mode="a")

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
                process_per_batch(to_predict_queue)
                for _ in range(len(tile_pred_cache)): tile_pred_cache.popitem()

                



class test_model():

    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return x
    
    def batch_size(self):
        return 12
    
    def min_patch_size(self):
        return 128

if __name__ == "__main__":

    in_file = "../DATA_SANDBOX/lxFlowAlign/data/train_data/paris_ortho1_rooftop/flow.tif"
    c_r = RasterDataset(in_file)
    mdl = test_model()
    out_file = "../DATA_SANDBOX/out_file.tif"
    c_r.predict_to_file(out_file, mdl, augmentations=[Trans_Identity()]*5)
    pass