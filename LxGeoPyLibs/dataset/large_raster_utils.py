import os
import pyvips
import numpy as np

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