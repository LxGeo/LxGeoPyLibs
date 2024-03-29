from torch.utils.data import Dataset
try:
    from functools import cached_property
except:
    from backports.cached_property import cached_property
import copy

class SpatialyProjectedDataset(object):
    def __init__(self, crs):
        self._crs=crs
    
    @cached_property
    def crs(self):
        return self._crs

class LazySetupDataset:
    def __init__(self, fn=None):
        self.is_setup=False
        self.__set_lazysetup_fn(fn)
    
    def __set_lazysetup_fn(self, fn):
        self.lazysetup_fn=fn
    
    def setup(self):
        if not self.is_setup:
            if not self.lazysetup_fn:
                raise Exception("LazySetupDataset setup function is not set!")
            self.lazysetup_fn()
        self.is_setup=True

class BoundedDataset(SpatialyProjectedDataset):

    def __init__(self, bounds_geom, crs=None):
        self.bounds_geom = bounds_geom
        SpatialyProjectedDataset.__init__(self, crs)

class Pixelized2DDataset(object):
    def __init__(self, x_pixel_size, y_pixel_size):
        self.x_pixel_size=x_pixel_size
        self.y_pixel_size=y_pixel_size
    
    def pixel_to_spatial_unit(self, vals):
        if isinstance(vals, (list, tuple)):
            return (vals[0]*self.x_pixel_size , vals[1]*self.y_pixel_size)
        elif isinstance(vals, int):
            return (vals*self.x_pixel_size , vals*self.y_pixel_size)
        else:
            raise Exception(f"Unexpected argument of type {type(vals)}!")

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
        
