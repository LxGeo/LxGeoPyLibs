from LxGeoPyLibs.dataset.vector_dataset import VectorDataset, vectors_map
from functools import partial
import pygeos
import numpy as np
import torch
from LxGeoPyLibs.geometry.rasterizers.polygons_rasterizer import polygons_to_multiclass2, polygonsWA_to_displacment_map
from LxGeoPyLibs.dataset.common_interfaces import BoundedDataset, PixelizedDataset
from LxGeoPyLibs.dataset.raster_dataset import RasterDataset
from LxGeoPyLibs.dataset.patchified_dataset import PatchifiedDataset

class RasterizedVectorDataset(RasterDataset, PixelizedDataset):

    def __init__(self, vector_path:str, gsd, rasterization_method, pixel_patch_size, pixel_patch_overlap, bounds_geom=None,
        augmentation_transforms=None, preprocessing=lambda x:x):

        self.reference_vector = VectorDataset(vector_path=vector_path)
        PixelizedDataset.__init__(self, gsd, gsd)
        if not bounds_geom:
            bounds_geom = self.reference_vector.bounds_geom
        self.setup_patch_per_pixel(pixel_patch_size, pixel_patch_overlap, bounds_geom)
        self._gsd=gsd
        self.rasterization_method = partial(rasterization_method, gsd=gsd, crs=self.reference_vector.crs())
        self.preprocessing=preprocessing
    
    def __getitem__(self, idx):
        window_geom = self.patch_grid[idx]
        loaded_gdf = self.reference_vector._load_vector_features_window(window_geom)
        #transformed_geoms = vec = list(map(lambda x: pygeos.to_shapely(x), transformed_geoms))
        rasterized_vector = self.rasterization_method(loaded_gdf.geometry, window_geom)
        rasterized_vector=self.preprocessing(rasterized_vector)
        return torch.from_numpy(rasterized_vector).float()
    
    def __len__(self):
        return PatchifiedDataset.__len__(self)
    
    def gsd(self):
        return self._gsd
    
    def bounds(self):
        return self.reference_vector.bounds()
    
    def crs(self):
        return self.reference_vector.crs()
    
    def get_stacked_batch(self, input_to_stack):
        #return [torch.stack(input_to_stack)]
        return torch.stack(input_to_stack)
    
    