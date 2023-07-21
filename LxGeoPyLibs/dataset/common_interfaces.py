from torch.utils.data import Dataset
from functools import cached_property
import copy

class SpatialyProjectedDataset(object):
    def __init__(self, crs):
        self._crs=crs
    
    @cached_property
    def crs(self):
        return self._crs

class BoundedDataset(SpatialyProjectedDataset):

    def __init__(self, bounds_geom, crs=None):
        self.bounds_geom = bounds_geom
        SpatialyProjectedDataset.__init__(self, crs)

class Pixelized2DDataset(object):
    def __init__(self, x_pixel_size, y_pixel_size):
        self.x_pixel_size=x_pixel_size
        self.y_pixel_size=y_pixel_size

class CompositionDataset(object):

    def __init__(self, parent_dataset, copy_parent=True):
        self.parent_dataset = copy.copy(parent_dataset) if copy_parent else parent_dataset
    
    def get_top_dataset(self):
        if not isinstance(self.parent_dataset, CompositionDataset):
            return self.parent_dataset
        else:
            return self.parent_dataset.get_top_dataset()

class PreProcessedDataset(CompositionDataset):
    
    def __init__(self, parent_dataset ,preprocessing_callable):
        CompositionDataset.__init__(self, parent_dataset)
        self.preprocessing_callable = preprocessing_callable
    
    def __len__(self):
        return len(self.parent_dataset)
    
    def __getitem__(self, index):
        parent_item = self.parent_dataset.__getitem__(index)
        return self.preprocessing_callable(parent_item)

class AugmentedDataset(CompositionDataset):
    
    def __init__(self, parent_dataset, augmentations):
        CompositionDataset.__init__(self, parent_dataset)
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.parent_dataset) * len(self.augmentations)
    
    def __getitem__(self, idx):        
        parent_item_idx = idx // (len(self.augmentations))
        c_augmentation_idx = idx % (len(self.augmentations))
        c_augmentation = self.augmentations[c_augmentation_idx]
        
        parent_item = self.parent_dataset[parent_item_idx]
        return c_augmentation(parent_item)
        
