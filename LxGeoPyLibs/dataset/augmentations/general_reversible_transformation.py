from skimage.transform import rotate
from skimage import exposure
import numpy as np
import torch
from functools import partial
import copy
from LxGeoPyLibs.ppattern.exceptions import OutOfTypesException


class GeneralIrrevirsibleTransformation:
    
    def __call__(self, item, **kwargs):        
        return self.__fwd__(item, **kwargs)
    
    def __fwd__(self, *args):
        raise NotImplementedError

class GeneralRevrsibleTransformation:
    
    def __init__(self, fwdfn, rwdfn):
        self.__fwd__fn__ = fwdfn
        self.__rwd__fn__ = rwdfn
        
    def __call__(self, item, **kwargs):        
        if kwargs.get("reverse", False):
            return self.__rwd__(item, **kwargs)
        else:
            return self.__fwd__(item, **kwargs)
    
    def __fwd__(self, *args):
        return self.__fwd__fn__(*args)
    
    def __rwd__(self, *args):
        return self.__rwd__fn__(*args)

class MultipleItemRevrsibleTransformation(GeneralRevrsibleTransformation):
    
    def __init__(self, parent_transformation : GeneralRevrsibleTransformation, applicable_map_keys):
        self.parent_transformation=parent_transformation
        self.applicable_map_keys=applicable_map_keys
    
    def __fwd__(self, items_dict:dict):
        out_items_dict = {}
        for k,v in items_dict.items():
            if k in self.applicable_map_keys:
                out_items_dict[k] = self.parent_transformation.__fwd__(v)
            else:
                out_items_dict[k] = copy.deepcopy(v)
        return out_items_dict
    def __rwd__(self, items_dict):
        out_items_dict = {}
        for k,v in items_dict.items():
            if k in self.applicable_map_keys:
                out_items_dict[k] = self.parent_transformation.__rwd__(v)
            else:
                out_items_dict[k] = copy.deepcopy(v)
        return out_items_dict
    
    

class TransIdentity(GeneralRevrsibleTransformation):
    def identity(x):
        return x
    
    def __init__(self):        
        GeneralRevrsibleTransformation.__init__(self,
            fwdfn=TransIdentity.identity,
            rwdfn=TransIdentity.identity,
        )
    
class AlongPlaneTransformation:
    def __init__(self, spatial_dims):
        self.spatial_dims = spatial_dims

class AlongDimensionTransformation:
    def __init__(self, dimension_index):
        self.dimension_index = dimension_index

    
class Rotation2DTransformation(AlongPlaneTransformation):
    def __init__(self, spatial_dims=(-2, -1)):
        AlongPlaneTransformation.__init__(self, spatial_dims)
    
    def rot90_fn(item, times, axes=(-2,-1)):
        if isinstance(item, np.ndarray):
            return np.rot90(item, k=times, axes=axes)
        elif isinstance(item, torch.Tensor):
            return torch.rot90(item, k=times, dims=axes) 
        else:
            raise OutOfTypesException(item, (np.ndarray, torch.Tensor))

class TransRot90(Rotation2DTransformation, GeneralRevrsibleTransformation):
    
    def __init__(self, spatial_dims=(-2, -1)):
        Rotation2DTransformation.__init__(self, spatial_dims)
        GeneralRevrsibleTransformation.__init__(self,
            fwdfn=partial(Rotation2DTransformation.rot90_fn, times=1),
            rwdfn=partial(Rotation2DTransformation.rot90_fn, times=3),
        )
    
class TransRot180(Rotation2DTransformation, GeneralRevrsibleTransformation):
    
    def __init__(self, spatial_dims=(-2, -1)):
        Rotation2DTransformation.__init__(self, spatial_dims)
        GeneralRevrsibleTransformation.__init__(self,
            fwdfn=partial(Rotation2DTransformation.rot90_fn, times=2),
            rwdfn=partial(Rotation2DTransformation.rot90_fn, times=2),
        )
    
class TransRot270(Rotation2DTransformation, GeneralRevrsibleTransformation):
    
    def __init__(self, spatial_dims=(-2, -1)):
        Rotation2DTransformation.__init__(self, spatial_dims)
        GeneralRevrsibleTransformation.__init__(self,
            fwdfn=partial(Rotation2DTransformation.rot90_fn, times=3),
            rwdfn=partial(Rotation2DTransformation.rot90_fn, times=1),
        )
    

class Transflip(AlongDimensionTransformation, GeneralRevrsibleTransformation):
    
    def flip_fn(item, dimension_index):
        if isinstance(item, np.ndarray):
            return np.take(item, indices=np.arange(item.shape[dimension_index])[::-1], axis=dimension_index)
        elif isinstance(item, torch.Tensor):
            return torch.index_select( item, dimension_index, torch.arange(item.size(dimension_index)-1, -1, -1) )
        else:
            raise OutOfTypesException(item, (np.ndarray, torch.Tensor))
        
    def __init__(self, dimension_index):
        AlongDimensionTransformation.__init__(self, dimension_index)
        GeneralRevrsibleTransformation.__init__(self,
            fwdfn=partial(Transflip.flip_fn, dimension_index=self.dimension_index),
            rwdfn=partial(Transflip.flip_fn, dimension_index=self.dimension_index),
        )

"""class Transfliplr(AlongDimensionTransformation, GeneralRevrsibleTransformation):
    
    def fliplr_fn(item, dimension_index):
        if isinstance(item, np.ndarray):
            return np.take(item, indices=np.arange(item.shape[dimension_index])[::-1], axis=dimension_index)
            #return np.fliplr(item)
        elif isinstance(item, torch.Tensor):
            return torch.index_select( item, dimension_index, torch.arange(item.size(dimension_index)-1, -1, -1) )
            #return torch.fliplr(item) 
        else:
            raise OutOfTypesException(item, (np.ndarray, torch.Tensor))
        
    def __init__(self, dimension_index=-1):
        AlongDimensionTransformation.__init__(self, dimension_index)
        GeneralRevrsibleTransformation.__init__(self,
            fwdfn=partial(Transfliplr.fliplr_fn, dimension_index=self.dimension_index),
            rwdfn=partial(Transfliplr.fliplr_fn, dimension_index=self.dimension_index),
        )"""
    
    