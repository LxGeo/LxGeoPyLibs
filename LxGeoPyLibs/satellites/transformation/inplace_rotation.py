from enum import Enum
from functools import partial
import rasterio as rio
from affine import Affine
import numpy as np
import click

class inplaceRotationAngle(Enum):
    rot0 = 0
    rot90 = 1
    rot180 = 2
    rot270 = 3

def rotation_respective_origin_getter(in_rotation : inplaceRotationAngle):
    #Returns a functor that takes a rasterio profile and returns origin position
    def pixel_to_coords(raster_profile, position_getter):
        return rio.transform.xy(raster_profile["transform"], *position_getter(raster_profile))

    if (in_rotation == inplaceRotationAngle.rot0):
        return partial(pixel_to_coords, position_getter=lambda profile: (0,0) )
    elif(in_rotation == inplaceRotationAngle.rot90):
        return partial(pixel_to_coords, position_getter=lambda profile: (0, profile["width"]) )
    elif(in_rotation == inplaceRotationAngle.rot180):
        return partial(pixel_to_coords, position_getter=lambda profile: (profile["height"], profile["width"]) )
    elif(in_rotation == inplaceRotationAngle.rot270):
        return partial(pixel_to_coords, position_getter=lambda profile: (profile["height"], 0) )


def rotation_respective_profile(in_rotation : inplaceRotationAngle, in_profile: rio.profiles.Profile, keep_pixel_size: bool):
    """
    Returns an updated profile
    """
    out_profile = in_profile.copy()
    
    if not keep_pixel_size:
        rotation_pivot = rio.transform.xy(in_profile["transform"], in_profile["height"]/2, in_profile["width"]/2)
        temp_transform = out_profile["transform"]
        temp_transform = Affine.rotation(-90*in_rotation.value, rotation_pivot) * temp_transform
        xoff,yoff = rotation_respective_origin_getter(in_rotation)(in_profile)
        out_profile["transform"] = Affine(temp_transform.a, temp_transform.b, xoff, temp_transform.d, temp_transform.e, yoff)

    if (in_rotation == inplaceRotationAngle.rot90) | (in_rotation == inplaceRotationAngle.rot270):
        out_profile["height"], out_profile["width"] = out_profile["width"],out_profile["height"]
    return out_profile

from shapely.geometry import box
def get_extents(profile):
    return box(*rio.transform.array_bounds(profile["height"], profile["width"], profile["transform"]))


def inplace_rotation(rio_dataset : rio.DatasetReader, rotation_angle : inplaceRotationAngle, out_raster_path : str, keep_pixel_size :bool ):
    """
    Applies inplace rotation of a dataset by transforming matrix data & geotransfrom & size
    """
    # Preapre output profile
    out_profile = rotation_respective_profile(rotation_angle, rio_dataset.profile, keep_pixel_size)
    with rio.open(out_raster_path, "w", **out_profile) as target_dst:
        target_dst.update_tags(rot90_count=rotation_angle.value)
        target_dst.write( np.rot90(rio_dataset.read(), rotation_angle.value, axes=(1,2)) )
    

@click.command()
@click.argument('in_raster_path', type=click.Path(exists=True))
@click.argument('out_raster_path', type=click.Path())
@click.option("-ra", "--rotation_angle", type=click.Choice(inplaceRotationAngle.__members__), 
              callback=lambda c, p, v: getattr(inplaceRotationAngle, v) if v else None, default="rot90")
@click.option('--keep_pixel_size', '-kps', is_flag=True, help="Keeps original pixel size (image is not well positionned)")
def main(in_raster_path, out_raster_path, rotation_angle, keep_pixel_size):
    """
    Program that rotates raster data array and transforms respective geotransform to keep the raster in its right position.
    Useful for post prediction fusion, since CNN are invariant to rotation.
    """
    with rio.open(in_raster_path) as in_rio_dst:
        inplace_rotation(in_rio_dst, rotation_angle, out_raster_path, keep_pixel_size)
    
if __name__ == "__main__":
    #in_path = "/home/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/LA/oneAtlas_catalogs_extracted/PHR1A_acq20220728_del1070b795/ortho_t.tif"
    #out_path = "/home/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/LA/oneAtlas_catalogs_extracted/PHR1A_acq20220728_del1070b795/ortho90_t.tif"
    main()
