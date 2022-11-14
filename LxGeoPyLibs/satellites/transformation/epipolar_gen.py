import click
import pyvips
from LxGeoPyLibs import _logger
from LxGeoPyLibs.satellites.imd import IMetaData
from LxGeoPyLibs.satellites.formulas import compute_rotation_angle
import math
from LxGeoPyLibs.dataset.large_raster_utils import load_vips, vips2numpy, crop_image
import cv2
import numpy as np
from scipy.stats import gaussian_kde
import rasterio as rio


def lazy_ref_epipolar_creation(iref1, iref2, i_alt1, i_alt2, epi1, epi2, rotation_angle):
    """
    Creates refined epipolar images using feature descriptor matching.
    Returns displacment applied vertically to refine epipolar/
    """
    
    with rio.open(iref1) as dst:
        ref1_x_size = dst.transform[0]
        ref1_bounds = dst.bounds

    with rio.open(iref2) as dst:
        ref2_x_size = dst.transform[0]
        ref2_bounds = dst.bounds
    
    # resolution check
    if (ref1_x_size != ref2_x_size):
        _logger.warning("Images have different pixel size")
        return
    
    bounds_displacment = [ abs(c1-c2) for c1,c2 in zip(ref1_bounds, ref2_bounds) ]
    if any([ disp>ref1_x_size for disp in bounds_displacment ]):
        _logger.warning("Images don't have the same bounds!")
        return
    

    img1 = load_vips(iref1)
    img2 = load_vips(iref2)
    img1_rot = img1.rotate(rotation_angle)
    img2_rot = img2.rotate(rotation_angle)
    assert (img1_rot.width == img2_rot.width), "Different rotated images width"
    assert (img1_rot.height == img2_rot.height), "Different rotated images height"
    img_rot_center = ( img1_rot.width//2, img1_rot.height//2 )
    
    ### crop size within rotated images
    MAX_CROP_SIZE = 2000
    img1_rot_crop=crop_image(img1_rot, img_center=img_rot_center, window_size=(MAX_CROP_SIZE, MAX_CROP_SIZE)); img1_rot_crop=vips2numpy(img1_rot_crop)
    img2_rot_crop=crop_image(img2_rot, img_center=img_rot_center, window_size=(MAX_CROP_SIZE, MAX_CROP_SIZE)); img2_rot_crop=vips2numpy(img2_rot_crop)

    _, _, v_disp = estimate_epipolarity_properties(img1_rot_crop, img2_rot_crop)
    disp_sign = (v_disp>0)

    i_alt1_rot = load_vips(i_alt1).rotate(rotation_angle)
    i_alt2_rot = load_vips(i_alt2).rotate(rotation_angle)
    i_alt1_epi1 = i_alt1_rot.embed(0,abs(v_disp)*(not disp_sign), i_alt1_rot.width, i_alt1_rot.height+abs(v_disp), extend="background", background=0)
    i_alt1_epi2 = i_alt2_rot.embed(0,abs(v_disp)*(disp_sign), i_alt2_rot.width, i_alt2_rot.height+abs(v_disp), extend="background", background=0)
    i_alt1_epi1.write_to_file(epi1)
    i_alt1_epi2.write_to_file(epi2)
    
    return v_disp

def estimate_2d_displacement(im_array_1, im_array_2):
    """
    Estimate displacement of close features in 2d space of one band images.
    Args:
        im_array_<i>: Numpy array of shape [H,W] or [H,W,1]
    """
    assert im_array_1.shape == im_array_2.shape, "Image arrayss have different shapes!"
    assert len(im_array_1.shape)==2 or im_array_1.shape[-1]==1, "Only one band images are supported!"

    detecteor = cv2.AKAZE_create()
    kp1, des1 = detecteor.detectAndCompute(im_array_1,None)
    kp2, des2 = detecteor.detectAndCompute(im_array_2,None)
    bf = cv2.BFMatcher()
    matches = bf.match(des1,des2)

    # Filter matched features and keep only low displacement matched features
    MAX_DISP = 20
    MIN_DISP = 3
    f_matches = list(filter(lambda m: MIN_DISP < abs(kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0])<MAX_DISP and MIN_DISP < abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1])<MAX_DISP  ,matches))

    disps_col = []
    disps_row = []
    for m in f_matches:
        kpt = kp1[m.queryIdx]
        kpq = kp2[m.trainIdx]
        displacement_col = kpt.pt[0] - kpq.pt[0]
        displacement_row = kpt.pt[1] - kpq.pt[1]
        disps_col.append(displacement_col)
        disps_row.append(displacement_row)
    
    disps_col=np.array(disps_col)
    disps_row=np.array(disps_row)
    return disps_col, disps_row

def estimate_epipolarity_properties( iref1, iref2 ):
    """
    Estimate epipolarity angle and refinement displacment over two axes using matched close features between two images.
    """

    WINDOW_SIZE = 2000
    img1 = load_vips(iref1)
    img2 = load_vips(iref2)

    assert (img1.width == img2.width), "Different rotated images width"
    assert (img1.height == img2.height), "Different rotated images height"
    img_center = ( img1.width//2, img2.height//2 )

    img1_crop=crop_image(img1, img_center=img_center, window_size=(WINDOW_SIZE, WINDOW_SIZE)); img1_crop=vips2numpy(img1_crop)
    img2_crop=crop_image(img2, img_center=img_center, window_size=(WINDOW_SIZE, WINDOW_SIZE)); img2_crop=vips2numpy(img2_crop)
    # Adding bands dimension if needed
    if len(img1_crop.shape)==2: np.expand_dims(img1_crop, -1)
    if len(img2_crop.shape)==2: np.expand_dims(img2_crop, -1)

    # Estimate displacement across all bands
    disps_col, disps_row = zip(*[estimate_2d_displacement(img1_crop[:,:,band_idx], img2_crop[:,:,band_idx]) for band_idx in range(img1_crop.shape[-1])])
    disps_col=np.concatenate(disps_col); disps_row=np.concatenate(disps_row)
   
    # Density estimation
    disps_colrow = np.vstack([disps_col, disps_row])
    disps_colrow = gaussian_kde(disps_colrow)(disps_colrow)

    common_disp_col = np.average(disps_col, weights=disps_colrow)
    common_disp_row = np.average(disps_row, weights=disps_colrow)

    from LxGeoPyLibs.geometry.utils_numpy import angle_between
    disp_vec=np.array([common_disp_col, common_disp_row])
    horizental_vector = np.array([1,0])
    epipolarity_angle = -math.degrees(angle_between(disp_vec, horizental_vector))
    
    return epipolarity_angle, common_disp_col, common_disp_row


@click.command()
@click.option('-iref1', '--input_reference_image_1', required=True, type=click.Path(exists=True), help="First reference image file path")
@click.option('-iimd1', '--input_imd_1', type=click.Path(exists=True), help="First reference image imd file path")
@click.option('-iref2', '--input_reference_image_2', required=True, type=click.Path(exists=True), help="Second reference image file path")
@click.option('-iimd2', '--input_imd_2', type=click.Path(exists=True), help="Second reference image imd file path")
@click.option('-i_alt1', '--input_alternative_image_1', type=click.Path(exists=True),
 help="First image to rotate file path (Optional: if not provided input_reference_image_1 will be rotated)"
)
@click.option('-i_alt2', '--input_alternative_image_2', type=click.Path(exists=True),
 help="Second image to rotate file path (Optional: if not provided input_reference_image_2 will be rotated)"
 )
@click.option('-epi1', '--output_epi_image_1', required=True, type=click.Path(), help="First epipolar image output path")
@click.option('-epi2', '--output_epi_image_2', required=True, type=click.Path(), help="Second epipolar image output path")
def main(input_reference_image_1, input_imd_1, input_reference_image_2, input_imd_2, input_alternative_image_1, input_alternative_image_2, output_epi_image_1, output_epi_image_2):
    """
    A program to generate epipolar images using input_reference_images and respective IMD (optional).
    It can create epipolar out of alterantive input images using the option 'input_alternative_image_<i>' but the following requierments (Same size, Same geotransform, Same projection)
    """
    
    if input_imd_1 and input_imd_2:
        imd1_obj = IMetaData(input_imd_1)
        imd2_obj = IMetaData(input_imd_2)
        epipolarity_angle = math.degrees(compute_rotation_angle(imd1_obj.satAzimuth(), imd1_obj.satElevation(), imd2_obj.satAzimuth(), imd2_obj.satElevation()))
    else:
        _logger.warning("None or missing IMD data provded!")
        _logger.warning("Estimating epipolarity angle using close feature displacement!")
        epipolarity_angle,_,_ = estimate_epipolarity_properties(iref1=input_reference_image_1, iref2=input_reference_image_2)
    
    _logger.info(f"Epipolarity angle = {epipolarity_angle}")

    if not input_alternative_image_1: input_alternative_image_1=input_reference_image_1
    if not input_alternative_image_2: input_alternative_image_2=input_reference_image_2
    lazy_ref_epipolar_creation(input_reference_image_1, input_reference_image_2,
     input_alternative_image_1, input_alternative_image_2,
      output_epi_image_1, output_epi_image_2,
       epipolarity_angle
       )


    
if __name__ == "__main__":
    main()