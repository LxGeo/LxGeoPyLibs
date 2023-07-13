

from LxGeoPyLibs.dataset.raster_dataset import RasterDataset
from LxGeoPyLibs.dataset.vector_dataset import VectorDataset
from LxGeoPyLibs.geometry.rasterizers import rasterize_from_profile
from LxGeoPyLibs.geometry.utils_rio import extents_to_profile
import rasterio as rio
from rasterio.plot import reshape_as_image
from shapely.geometry import box
import geopandas as gpd
from LxGeoPyLibs.projection_utils import reproject_geom

from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.windows
def reproject_raster_to_image(raster, dst_crs, aoi):
    transform, width, height = calculate_default_transform(
        raster.crs, dst_crs, raster.width, raster.height, *raster.bounds, resolution=raster.transform[0])
    
    kwargs = raster.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".tif") as tmp:
        temp_file_path = tmp.name
        
        with rio.open(temp_file_path, 'w', **kwargs) as dst:

            reporjected_image = reproject(
                    source=rio.band(raster, 1),
                    destination=rio.band(dst, 1),
                    src_transform=raster.transform,
                    src_crs=raster.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
            
        with rio.open(temp_file_path) as dst_r:
            window = rasterio.windows.from_bounds(*aoi, dst_r.transform)
            return reshape_as_image(dst_r.read(window=window))[:,:,0]
    return reporjected_image

def get_binary_map(fp, aoi, gsd=0.2985821, dst_crs=None):
    if fp.lower().endswith(".shp"):
        gdf = gpd.read_file(fp)
        crs=gdf.crs
        if dst_crs and dst_crs!=crs:
            gdf=gdf.to_crs(dst_crs)
        rasterization_profile = extents_to_profile(aoi, gsd=gsd, crs=gdf.crs, count=1, dtype=rio.uint8)
        binary_map = rasterize_from_profile(gdf.geometry, rasterization_profile, 1)
    else:
        with rio.open(fp) as dst:
            if dst_crs and dst_crs!=dst.crs:
                binary_map = reproject_raster_to_image(dst, dst_crs, aoi)
            else:
                window = rasterio.windows.from_bounds(*aoi, dst.transform)
                binary_map = dst.read(window=window)
            crs = dst.crs
    return binary_map, crs

def get_aoi(fp, dst_crs=None):
    if fp.lower().endswith(".shp"):
        gdf = gpd.read_file(fp)
        crs=gdf.crs
        bounds = gdf.total_bounds
        if dst_crs:
            bounds = reproject_geom(box(*bounds), gdf.crs, dst_crs).bounds
    else:
        with rio.open(fp) as dst:
            bounds = dst.bounds
            crs=dst.crs
            if dst_crs:
                bounds = reproject_geom(box(*bounds), dst.crs, dst_crs).bounds
    if dst_crs:
        return bounds, dst_crs
    else:
        return bounds, crs 

def get_iou(im1, im2):
    overlap = im1.astype(bool)*im2.astype(bool) # Logical AND
    union = im1.astype(bool) + im2.astype(bool) # Logical OR
    IOU = overlap.sum()/float(union.sum())
    return IOU

if __name__ == "__main__":

    ref_map_path = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/India_Mumbai_A_Neo/h_India_Mumbai_A_Neo_.shp"
    subject_map_path = "C:/DATA_SANDBOX/Alignment_Project/alignment_results/multistage/India_Mumbai_gt/sgbm+tm/v2.shp"
    #subject_map_path = "C:/DATA_SANDBOX/Alignment_Project/alignment_results/lxProximityAlign1D/India_Mumbai_gt_v1tov2/aligned.shp"
    #subject_map_path = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Brazil_Vila_Velha_A_Neo/Brazil_Vila_Velha_A_Neo.shp"
    #subject_map_path = "C:/DATA_SANDBOX/Alignment_Project/alignment_results/lxProximityAlign2D/Ethiopia_Addis_Ababa_gt_v2tov1/aligned.shp"

    ref_bounds, ref_crs = get_aoi(ref_map_path)
    subject_bounds, subject_crs = get_aoi(subject_map_path, ref_crs)
    common_aoi = box(*ref_bounds).intersection(box(*subject_bounds)).bounds

    ref_binary,ref_crs = get_binary_map(ref_map_path, common_aoi)
    subject_binary, subject_crs = get_binary_map(subject_map_path, common_aoi, dst_crs=ref_crs)

    iou_value = get_iou(ref_binary, subject_binary)
    print(f"iOU: {iou_value}")


