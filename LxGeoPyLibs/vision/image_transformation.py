# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:50:07 2022

@author: cherif
"""

from skimage.transform import rotate
from skimage import exposure
import numpy as np
import torch

rot90_fn = lambda x, times: np.rot90(x, k=times, axes=(1, 2)) if isinstance(x, np.ndarray) else torch.rot90(x, k=times, dims=(1, 2)) 

class Image_Transformation:
    
    def __call__(self, *args, **kwargs):
        
        to_return = []
        for c_arg in args:
            if isinstance(c_arg, (np.ndarray,torch.Tensor)):
                if kwargs.get("reverse", False):
                    to_return.append(self.__rwd__(c_arg)[0])
                else:
                    to_return.append(self.__fwd__(c_arg)[0])
            else:
                to_return.append( self.__call__(*c_arg) )
        return to_return
        """
        if kwargs.get("reverse", False):
            return self.__rwd__(*args)
        else:
            return self.__fwd__(*args)"""
    
    def __fwd__(self, *args):
        raise NotImplementedError
    
    def __rwd__(self, *args):
        raise NotImplementedError
        

class Trans_Identity(Image_Transformation):
    
    def __fwd__(self, *args):
        return args
    def __rwd__(self, *args):
        return args
    
class Trans_Rot90(Image_Transformation):
    
    def __fwd__(self, *args):
        return tuple(rot90_fn(c_arg,1) for c_arg in args)
    def __rwd__(self, *args):
        return Trans_Rot270()(*args)
    

class Trans_Rot180(Image_Transformation):
    
    def __fwd__(self, *args):
        return tuple(rot90_fn(c_arg,2) for c_arg in args)
    def __rwd__(self, *args):
        return self.__fwd__(*args)

class Trans_Rot270(Image_Transformation):
    
    def __fwd__(self, *args):
        return tuple(rot90_fn(c_arg,3) for c_arg in args)
    def __rwd__(self, *args):
        return Trans_Rot90()(*args)
        
        
flipud_fn = lambda x: np.flipud(x) if isinstance(x, np.ndarray) else torch.flipud(x) 
class Trans_Flipud(Image_Transformation):
    
    def __fwd__(self, *args):
        return tuple(np.flipud(c_arg) for c_arg in args)
    def __rwd__(self, *args):
        return self.__fwd__(args)
    

fliplr_fn = lambda x: np.fliplr(x) if isinstance(x, np.ndarray) else torch.fliplr(x)
class Trans_fliplr(Image_Transformation):
    
    def __fwd__(self, *args):
        return tuple(np.fliplr(c_arg) for c_arg in args)
    def __rwd__(self, *args):
        return self.__fwd__(args)
    
class Trans_gaussian_noise(Image_Transformation):
    def __call__(self, image1, gt1):
        noise = np.random.normal(loc = 0.0, scale = 2, size = image1.shape) 
        image1_t = np.clip(image1+noise, 0 ,255).astype(image1.dtype)
        return (image1_t, gt1)

class Trans_gamma_adjust(Image_Transformation):    
    def __init__(self, gamma=1.5):
        self.gamma = gamma
        
    def __call__(self, image1, gt1):
        image1_t = exposure.adjust_gamma(image1/255, self.gamma)*255        
        return (image1_t, gt1)

class Trans_equal_hist(Image_Transformation):    
    def __call__(self, image1, gt1):
        image1_t = (exposure.equalize_hist(image1)*255).astype(image1.dtype)
        return (image1_t, gt1)

class Trans_contrast_stretch(Image_Transformation):    
    def __call__(self, image1, gt1):
        p2, p98 = np.percentile(image1, (2, 98))
        image1_t = exposure.rescale_intensity(image1, in_range=(p2, p98))*255
        return (image1_t, gt1)
