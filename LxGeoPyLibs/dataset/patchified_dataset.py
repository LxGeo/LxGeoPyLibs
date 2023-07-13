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
                                    filter_predicate = lambda x: pygeos.intersects(pygeos.envelope(x), bounds_geom) )
        
        self.is_setup=True
    
    def __len__(self):
        return len(self.patch_grid)
    
    def __getitem__(self, index):
        return self.patch_grid[index]
    
    def gsd(self):
        raise NotImplementedError

    def get_stacked_batch(self, input_to_stack:[Any]):
        raise NotImplementedError


class CallableModel():

    def __init__(self,callable=lambda x:x, bs=1, mps=128):
        super().__init__()
        self.bs=bs
        self.mps = mps
        self.forward = callable
        self.device="cpu"

    def __call__(self, x):
        return self.forward(x)
    
    def batch_size(self):
        return self.bs
    
    def min_patch_size(self):
        return self.mps
    
    def predict_step(self, batch, batch_idx: int = None, dataloader_idx: int = 0):
        return self(batch)
