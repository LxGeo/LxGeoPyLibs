from LxGeoPyLibs.dataset.vector_dataset import VectorDataset, vectors_map
from functools import partial
import pygeos
import numpy as np
import torch


class RasterizedVectorDataset(VectorDataset):

    def __init__(self, vector_path:str, augmentation_transforms=None, preprocessing=None,
     bounds_geom=None, spatial_patch_size=None, spatial_patch_overlap=None,
    rasterization_method=None):

        super().__init__(vector_path=vector_path, augmentation_transforms=augmentation_transforms,
         preprocessing=preprocessing, bounds_geom=bounds_geom, spatial_patch_size=spatial_patch_size,
         spatial_patch_overlap=spatial_patch_overlap)
        
        self.rasterization_method = partial(rasterization_method, crs=super().crs())
    
    def __getitem__(self, idx):
        transformed_geoms = super().__getitem__(idx)
        transformed_geoms = vec = list(map(lambda x: pygeos.to_shapely(x), transformed_geoms))
        window_geom = super(VectorDataset, self).__getitem__(idx)
        rasterized_vector = self.rasterization_method(transformed_geoms, window_geom)
        return torch.from_numpy(rasterized_vector).float()
    
    def get_stacked_batch(self, input_to_stack):
        return [torch.stack(input_to_stack)]
    
    