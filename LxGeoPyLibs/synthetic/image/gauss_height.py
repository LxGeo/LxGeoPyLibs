
from functools import partial
from tkinter import N
from typing import DefaultDict
import pyvips
from LxGeoPyLibs.dataset.large_raster_utils import vips2numpy
import numpy as np
from skimage import transform
from skimage.transform import AffineTransform,ProjectiveTransform
import tqdm
import random


MASK_SANITY=5000

def random_homography():
    """
    Returns a transformer object to warp images.
    """
    def random_shear():
        """
        """
        return AffineTransform(shear=np.random.rand()*np.pi/3)    
    def random_projective():
        """
        """
        projective_matrix = np.array([[np.random.normal(1,0.01), np.random.normal(0,0.1), 0],
                   [np.random.normal(0,0.1), np.random.normal(1,0.01), 0],
                   [0.0015, 0.0015, 1]])
        return ProjectiveTransform(matrix=projective_matrix)    
    return np.random.choice([random_shear, random_projective])()

def generate_gauss_height_map(size, initial_value=0, dims_blobs_scaler=(10,10), blobs_size_normal_dist=None):
    """
    """

    out_map = initial_value * np.ones(size)

    # number of blobs across both axes
    N_blobs_x = size[0] // dims_blobs_scaler[0]; N_blobs_y = size[1] // dims_blobs_scaler[1];
    total_blob_num = N_blobs_x * N_blobs_y

    # blob position within the size of the image
    blobs_position = np.random.randint(low = (0,0), high=size, size=[total_blob_num,2] )

    # blob size distribution setting
    if not blobs_size_normal_dist:
        blobs_size = np.random.normal(size[0]/100, 10, size=total_blob_num)
    else:
        blobs_size = blobs_size_normal_dist(size=total_blob_num)
    # transform blobs_size to odd numbers
    blobs_size = (blobs_size // 2 * 2 + 1).astype(int)

    ## Generating blobs first
    random_blob = lambda b_size : rectangular_gauss_map(b_size, np.random.normal(b_size//5, b_size//100))
    blobs_cache = DefaultDict(list)
    for b_size in np.unique(blobs_size):
        gauss_rect = random_blob(b_size)
        for t in [random_homography(), random_homography(), random_homography()]:
            blobs_cache[b_size].append(transform.warp(gauss_rect, t.inverse))

    for blob_pos, blob_size in tqdm.tqdm(zip(blobs_position, blobs_size), desc="Outmap filling!", total =total_blob_num):
        blob_radius = (blob_size-1) //2        
        out = random.choice(blobs_cache[blob_size])

        start_x = blob_pos[0] - blob_radius; start_y = blob_pos[1] - blob_radius;
        end_x = start_x + blob_size; end_y = start_y + blob_size;
        
        if start_x<0: out=out[-start_x:,:]; start_x=0
        if start_y<0: out=out[:,-start_y:]; start_y=0;
        if end_x>size[0]: out=out[: size[0]-end_x,:]; end_x=size[0];
        if end_y>size[1]: out=out[:,:size[1]-end_y]; end_y=size[1];

        add_sign = random.choice([1,-1])
        out_map[start_x: start_x+out.shape[0], start_y: start_y+out.shape[1]] += add_sign*out

    heighest_value = max(out_map.max(), -out_map.min())
    out_map /= heighest_value
    return out_map

def rectangular_gauss_map(side_length, sigma):
    """
    Generates a gaussian height centered image.
    Args:
        side_length: odd integer equal to the size of a side
        sigma: float defining the extent of the gaussian. Given sigma, the lowest pixel value will be equal = expon( -((side_length-1)/2)**2 / sigma**2 )
    """
    assert side_length < MASK_SANITY, "Rectangle size is too big!, choose lower than {}".format(MASK_SANITY)
    assert side_length % 2, "Rectanle size should be odd!"
    max_x = ( side_length - 1 ) // 2
    value = np.exp(-(max_x**2)/(2*sigma**2))
    im_gaussmat=vips2numpy(pyvips.Image.gaussmat(sigma,value, precision=pyvips.Precision.FLOAT))[:,:,0]
    return im_gaussmat

if __name__ == "__main__":
    blobs_size_normal_dist = partial(np.random.normal, 200,10)
    dims_blobs_scaler = (5,5)
    out_map = generate_gauss_height_map((1000,1000), dims_blobs_scaler=dims_blobs_scaler, blobs_size_normal_dist=blobs_size_normal_dist)
    pass


