import os
import pyvips
import numpy as np
from typing import Union

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# map np dtypes to vips
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}


def load_vips(image_descriptor):
    """
    Function to load pyvips image from descriptor.
    image_descriptor: Union[ str, numpy_array, pyvips_image]
    """
    if type(image_descriptor) == str :
        assert os.path.exists(image_descriptor), "Image not found!"
        vips_image = pyvips.Image.new_from_file(image_descriptor)
    elif type(image_descriptor) == np.ndarray:
        vips_image = numpy2vips(image_descriptor)
    elif type(image_descriptor) == pyvips.Image:
        vips_image = pyvips.Image.new_from_image(image_descriptor)
    else:
        raise Exception("Not recognized image descriptor of type {}".format(type(image_descriptor)))    
    return vips_image


def numpy2vips(a):
    height, width, bands = a.shape
    linear = a.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      dtype_to_format[str(a.dtype)])
    return vi


# vips image to numpy array
def vips2numpy(vi):
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])

def rotate_large_array_to_file(in_raster_descriptor, rotation_angle, out_raster_descriptor):
    """
    Rotate tif files
    """
    vips_image = numpy2vips(in_raster_descriptor)
    rot_vips_image = vips_image.rotate(rotation_angle)
    rot_vips_image.write_to_file(out_raster_descriptor)
    return

def rotate_large_raster(raster_descriptor, rotation_angle):
    """
    Args:
        raster_descriptor could be:
            str -> path of raster to rotate
            ndarray -> numpy array
        rotation_angle: double
    Returns:
        numpy array
    """
    vips_image =load_vips(raster_descriptor)
    
    """
    forward_translation_mat = Affine.translation(*rotation_center)
    rotation_mat = Affine.rotation(rotation_angle)
    backward_translation_mat = Affine.translation(-rotation_center[0], -rotation_center[1])
    combined_mat = forward_translation_mat * rotation_mat * backward_translation_mat
    vips_transform_matrix = (combined_mat.a, combined_mat.b, combined_mat.d, combined_mat.e)
    rot_vips_image = vips_image.affine(vips_transform_matrix, idx=combined_mat.xoff, idy=combined_mat.yoff)   
    """

    rot_vips_image = vips_image.rotate(rotation_angle)    
    return vips2numpy(rot_vips_image)

def crop_image(img: Union[np.ndarray, pyvips.Image], img_center=None, window_size=(2000,2000)):
    """
    Centerd image crop function.
    Args:
        img_center: Tuple of numerical values
    """
    if img_center is None:
        img_center = ( img.width//2, img.height//2 )
    else:
        assert len(img_center)==2, "Image center should be a tuple of two numerical values"
        assert int(img_center[0])==img_center[0] and int(img_center[1]) == img_center[1], "image center should be integer!"
    crop_col_start, crop_col_end = max(0, img_center[0]-window_size[0]), min(img.width, img_center[0]+window_size[0])
    crop_row_start, crop_row_end = max(0, img_center[1]-window_size[1]), min(img.height, img_center[1]+window_size[1])

    if type(img)==np.ndarray:
        return img[crop_col_start:crop_col_end, crop_row_start:crop_row_end]
    elif type(img)==pyvips.Image:
        return img.crop( crop_col_start, crop_row_start, crop_col_end-crop_col_start, crop_row_end-crop_row_start )
    else:
        raise Exception(f"Type {type(img)} is not supported for cropping!")