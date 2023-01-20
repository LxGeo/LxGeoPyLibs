
import typing
from collections import OrderedDict
from LxGeoPyLibs.dataset.patchified_dataset import PatchifiedDataset
import pygeos
import torch

class HeteroDataset(PatchifiedDataset):

    def __init__(self, sub_datasets_init : typing.OrderedDict[str, dict], spatial_patch_size=None, spatial_patch_overlap=None, bounds_geom=None):
        
        self.sub_datasets = OrderedDict()
        for dst_id, dst_definition in sub_datasets_init.items():
            dataset_type = dst_definition.pop("dataset_type")
            self.sub_datasets[dst_id] = dataset_type(**dst_definition)

        common_area_geom = pygeos.intersection_all(
            [pygeos.box(*c_dataset.bounds()) for c_dataset in self.sub_datasets.values()]
            )
        
        if pygeos.is_empty(common_area_geom):
            raise Exception("Area mismatch.")
        
        if bounds_geom:
            assert not pygeos.is_empty( pygeos.intersection(common_area_geom, bounds_geom) ), "bounds_geom doesn't intersect with common area"
            self.bounds_geom = bounds_geom
        else:
            self.bounds_geom=common_area_geom
        
        if not None in (spatial_patch_size, spatial_patch_overlap):
            self.setup_spatial(spatial_patch_size, spatial_patch_overlap, self.bounds_geom)
        
    
    def setup_spatial(self, patch_size, patch_overlap, bounds_geom=None):

        self.patch_size= patch_size
        self.patch_overlap= patch_overlap
        
        # If no bounds provided, use image bounds
        if not bounds_geom:
            bounds_geom = self.bounds_geom

        pixel_x_size = self.gsd()
        pixel_y_size = self.gsd()

        patch_size_spatial = (self.patch_size[0]*pixel_x_size, self.patch_size[1]*pixel_y_size)
        patch_overlap_spatial = self.patch_overlap*pixel_x_size

        if not bounds_geom:
            bounds_geom=self.bounds_geom
        super().__init__(patch_size_spatial, patch_overlap_spatial, bounds_geom)
        for dst in self.sub_datasets.values():
            dst.setup_spatial(patch_size_spatial, patch_overlap_spatial, bounds_geom)
    
    
    def __getitem__(self, idx):
        ## Temporary getitem
        
        assert self.is_setup, "Dataset is not set up!"
        window_idx = idx        
        window_geom = super().__getitem__(window_idx)                
        
        sub_items = []
        for k, dst in self.sub_datasets.items():
            sub_items.append(dst[window_idx])
                
        return tuple(sub_items)
    
    def get_stacked_batch(self, input_to_stack):

        dezipped = list(zip(*input_to_stack))

        return [ torch.stack(d) for d in dezipped ]
    
    def gsd(self):
        return 0.5
    
    def crs(self):
        return list(self.sub_datasets.values())[0].crs()

