

import os
import json
from osgeo import gdal
import geopandas as gpd
import rasterio as rio

def get_full_path_from_id(conf_path, file_id):
    with open(conf_path) as f:
        conf_content = json.load(f)    
    return os.path.join(os.path.dirname(conf_path), conf_content[file_id])


def reproject_view_from_file(conf_file_path):

    with open(conf_file_path) as f:
        v_conf = json.load(f)
    
    dst_epsg_code = f"EPSG:{v_conf['utm_code']}"

    # warp raster
    src_epsg_code=None
    
    in_path = get_full_path_from_id(conf_file_path, "ortho")
    out_path = os.path.join( os.path.dirname(in_path), "utm_"+os.path.basename(in_path) )
    with rio.open(in_path) as dataset:
        src_epsg_code = str(dataset.crs)
    
    if (src_epsg_code!=dst_epsg_code):
        gdal.Warp(out_path,in_path,dstSRS=dst_epsg_code)
        os.replace(out_path, in_path)

    # warp vector
    in_path = get_full_path_from_id(conf_file_path, "buildings_vector")
    #out_path = os.path.join( os.path.dirname(in_path), "utm_"+os.path.basename(in_path) )
    in_gdf = gpd.read_file(in_path)
    in_gdf=in_gdf[in_gdf.is_valid]
    if (str(in_gdf.crs).lower()!=dst_epsg_code.lower()):
        in_gdf.to_crs(dst_epsg_code).to_file(in_path)

    

if __name__ == "__main__":
    view_dir = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Sweden_Stockholm_B_Neo/"
    conf_file_path1 = view_dir+"file_conf.json"
    conf_file_path2 = view_dir+"file_conf_gt.json"
    if os.path.isfile(conf_file_path1):
        reproject_view_from_file(conf_file_path1)

    if os.path.isfile(conf_file_path2):
        reproject_view_from_file(conf_file_path2)
