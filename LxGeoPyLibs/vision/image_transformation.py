# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:50:07 2022

@author: cherif
"""

from skimage.transform import rotate
from skimage import exposure
import numpy as np

class Image_Transformation:
    
    def __call__(image,gt, *args):
        pass

class Trans_Identity(Image_Transformation):
    def __call__(self, image1, gt1, *args):        
        return image1, gt1, args
    
class Trans_Rot90(Image_Transformation):
    def __call__(self, image1, gt1, *args):
        image1_t = rotate(image1, 90)
        gt1_t = rotate(gt1, 90)
        
        return (image1_t, gt1_t, *(rotate(c_arg,90) for c_arg in args))

class Trans_Rot180(Image_Transformation):
    def __call__(self, image1, gt1, *args):
        image1_t = rotate(image1, 180)
        gt1_t = rotate(gt1, 180)
        
        return (image1_t, gt1_t, *(rotate(c_arg,180) for c_arg in args))

class Trans_Rot270(Image_Transformation):
    def __call__(self, image1, gt1, *args):
        image1_t = rotate(image1, 270)
        gt1_t = rotate(gt1, 270)
        
        return (image1_t, gt1_t, *(rotate(c_arg,270) for c_arg in args))

class Trans_Flipud(Image_Transformation):
    def __call__(self, image1, gt1, *args):
        image1_t = np.flipud(image1)
        gt1_t = np.flipud(gt1)
        
        return (image1_t, gt1_t, *(np.flipud(c_arg) for c_arg in args))
    
class Trans_fliplr(Image_Transformation):
    def __call__(self, image1, gt1, *args):
        image1_t = np.fliplr(image1)
        gt1_t = np.fliplr(gt1)
        
        return (image1_t, gt1_t, *(np.fliplr(c_arg) for c_arg in args))

class Trans_gaussian_noise(Image_Transformation):
    def __call__(self, image1, gt1, *args):
        noise = np.random.normal(loc = 0.0, scale = 2, size = image1.shape) 
        image1_t = np.clip(image1+noise, 0 ,255).astype(image1.dtype)
        return (image1_t, gt1, args)

class Trans_gamma_adjust(Image_Transformation):    
    def __init__(self, gamma=1.5):
        self.gamma = gamma
        
    def __call__(self, image1, gt1, *args):
        image1_t = exposure.adjust_gamma(image1/255, self.gamma)*255        
        return (image1_t, gt1, args)

class Trans_equal_hist(Image_Transformation):    
    def __call__(self, image1, gt1, *args):
        image1_t = (exposure.equalize_hist(image1)*255).astype(image1.dtype)
        return (image1_t, gt1, args)

class Trans_contrast_stretch(Image_Transformation):    
    def __call__(self, image1, gt1, *args):
        p2, p98 = np.percentile(image1, (2, 98))
        image1_t = exposure.rescale_intensity(image1, in_range=(p2, p98))*255
        return (image1_t, gt1, args)
