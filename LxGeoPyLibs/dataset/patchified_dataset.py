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
from LxGeoPyLibs.dataset.common_interfaces import BoundedDataset
import geopandas as gpd

class PatchifiedDataset(BoundedDataset):
    """
    
    """

    def __init__(self, spatial_patch_size, spatial_patch_overlap, bounds_geom, crs=None):
        super().__init__(bounds_geom)

        """
        Setup patch loading settings using spatial coordinates.
        Args:
            patch_size: a couple of positive integers in spatial metric.
            patch_overlap: a positive integer in spatial metric.
            bounds_geom: pygeos polygon
        """
        
        BoundedDataset.__init__(self, bounds_geom, crs)
        
        # TODO fix following line (it's not totally correct)
        if type(spatial_patch_overlap) in (tuple, list):
           spatial_patch_overlap = spatial_patch_overlap[0] 

        self.spatial_patch_size = spatial_patch_size
        self.spatial_patch_overlap = spatial_patch_overlap
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
    
    def patches_gdf(self):
        return gpd.GeoDataFrame(geometry=self.patch_grid, crs=self.crs)


class PixelPatchifiedDataset(PatchifiedDataset):

    def __init__(self, pixel_x_size, pixel_y_size):
        self.pixel_x_size=pixel_x_size
        self.pixel_y_size=pixel_y_size

    ### should be fixed to meters not pixels
    def setup_patch_per_pixel(self, pixel_patch_size, pixel_patch_overlap, bounds_geom):
        """
        Setup patch loading settings using spatial coordinates.
        Args:
            patch_size: a tuple of positive integers in pixels.
            patch_overlap: a positive integer in pixels.
            bounds_geom: pygeos polygon
        """
        self.pixel_patch_size= pixel_patch_size
        self.pixel_patch_overlap= pixel_patch_overlap

        patch_size_spatial = (self.pixel_patch_size[0]*self.pixel_x_size, self.pixel_patch_size[1]*self.pixel_y_size)
        patch_overlap_spatial = self.pixel_patch_overlap*self.pixel_x_size

        PatchifiedDataset.__init__(self, patch_size_spatial, patch_overlap_spatial, bounds_geom)

class CallableModel():

    def __init__(self, bs=1, mps=128, functor=None):
        super().__init__()
        self.bs=bs
        self.mps = mps
        self.functor=functor
        #self.device="cpu"

    def __call__(self, x):
        if self.functor:
            return self.functor(x)
        return self.forward(x)
    
    def batch_size(self):
        return self.bs
    
    def min_patch_size(self):
        return self.mps
    
    def predict_step(self, batch, batch_idx: int = None, dataloader_idx: int = 0):
        return self(batch)
