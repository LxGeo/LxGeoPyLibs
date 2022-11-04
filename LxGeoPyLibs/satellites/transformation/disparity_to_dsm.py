from LxGeoPyLibs.dataset.large_raster_utils import load_vips, vips2numpy
import pyvips
import click
import rasterio as rio
from rasterio.plot import reshape_as_raster

def disparity_to_dsm(disparity_map, inv_rotation_angle, v_displacement, out_dsm_path, profile, scale_factor=1/10 ):
    """
    Apply reverse rotation of disparity map over respective image (epipolar 1 or 2)
    Args:
        disparity_map: image descriptor. Union [path string, numpy array]
        inv_rotation_angle: minus angle of epipolarity
        v_displacement: vertical displacement used before to refine epipolarity
        out_dsm_path: output dsm path
        profile: used to save geotiff (geotransform,...)
        dsm_over_ref: boolean used to generate dsm over reference or target image rooftop.
    """
    disparity_map = load_vips(disparity_map)
    disp_sign = (v_displacement>0)
    disparity_map_reversed = disparity_map.crop(
        0,abs(v_displacement)*(not disp_sign),
        disparity_map.width, disparity_map.height-abs(v_displacement)
        ).rotate(
            inv_rotation_angle, interpolate=pyvips.vinterpolate.Interpolate.new('nearest')
            ).linear(
                scale_factor,
                0
            )
    start_h = (disparity_map_reversed.height//2 - profile['height']//2 )
    start_w = (disparity_map_reversed.width//2 - profile['width']//2 )    
    disparity_map_reversed = disparity_map_reversed.crop(start_w, start_h, profile['width'], profile['height'])
    profile["dtype"] = rio.float32; profile["nodata"]=None; profile["count"]=1 
    with rio.open(out_dsm_path, "w", **profile) as tar:
        tar.write(reshape_as_raster(vips2numpy(disparity_map_reversed)))


@click.command()
@click.option('-idisp', '--input_disparity', required=True, type=click.Path(exists=True), help="Path to input disparity map")
@click.option('-ro', '--ref_ortho', required=True, type=click.Path(exists=True), help="Path to original ortho image (not epipolar)")
@click.option('-ira', '--inverse_rotation_angle', required=True, type=click.FLOAT, help="Rotation angle to use to inverse epipolarity")
@click.option('-rvd', '--refinement_vertical_disp', required=True, type=click.INT, help="Vertical displacement used to refine epipolar")
@click.option('-o', '--output_path', required=True, type=click.Path(), help="Path to output dsm map")
@click.option('-sf', '--scale_factor', default=1/10, type=click.FLOAT, help="Scale factor used to turn disp into meters")
def main(input_disparity, ref_ortho, inverse_rotation_angle, refinement_vertical_disp, output_path, scale_factor):
    
    with rio.open(ref_ortho) as dst:
        out_dsm_profile = dst.profile.copy()
        out_dsm_profile.update({"count":1, "dtype":rio.float32})
    # disparity scale factor: (1/10) output disparity are in tenth of pixels & resolution is mandatory to get heights in meter
    disparity_scale_factor = 1/10; disparity_scale_factor /= out_dsm_profile["transform"].a
    
    disparity_to_dsm(input_disparity, inverse_rotation_angle, refinement_vertical_disp, output_path, out_dsm_profile, scale_factor)
    

if __name__ == "__main__":
    main()