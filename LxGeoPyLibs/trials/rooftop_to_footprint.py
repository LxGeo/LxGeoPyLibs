
from LxGeoPyLibs.dataset.vector_dataset import VectorDataset
from LxGeoPyLibs.dataset.w_vector_dataset import WVectorDataset
import geopandas as gpd
import warnings;   warnings.filterwarnings("ignore")
import pandas as pd
import tqdm
import numpy as np
import typing
import shapely
from functools import partial

def baseToRoof(row, dDx, dDy, height_column="AGL", inverse=False):
    """
    Apply affine transformation on a geometry row based on its height attribute
    Args:
        row: geopandas row containing geometry and height attribute
        height_column: Column name with height values
        dDx: displacement accross x axis
        dDy: displacement accross y axis
        inverse: (bool) if true apply roofToBase instead.
    return:
        row :shapely geometry of shifted polygon
    """
    height = row[height_column]    
    transformMatrix = [1, 0, 0, 1, -dDx*height, -dDy*height]
    # Roof to base
    if inverse:
        transformMatrix = [1, 0, 0, 1, dDx*height, dDy*height]    
    row["geometry"] = shapely.affinity.affine_transform(row["geometry"], transformMatrix)        
    return row["geometry"]

def move_rooftop_to_footprint(in_vector_path, out_vector_path, disp_constants: typing.Tuple[float, float], height_column):
    """
    """
    in_dst = VectorDataset(in_vector_path, spatial_patch_size=(512,512), spatial_patch_overlap=0)
    out_dst = WVectorDataset(out_vector_path, schema=in_dst.fio_dataset().schema,crs=in_dst.crs())
    transform_fn = partial(baseToRoof, dDx=disp_constants[0], dDy=disp_constants[1], height_column=height_column)

    for c_window in tqdm.tqdm(in_dst.patch_grid, desc="Window processing "):
        in_view = in_dst._load_vector_features_window(c_window)
        in_view["geometry"] = in_view.apply(lambda row: transform_fn(row), axis=1)
        out_dst.add_feature(in_view)
    return

if __name__ == "__main__":

    in_vector_path = "/home/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/paris_tristereo_dhm_plus_external/alignment2/approach3/PHR1A_acq20180326_del736ec042/aligned.shp"
    out_vector_path = "/home/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/paris_tristereo_dhm_plus_external/alignment2/approach3/PHR1A_acq20180326_del736ec042/base.shp"
    disp_constants = (0,0)
    height_column = "al_height"
    move_rooftop_to_footprint(in_vector_path, out_vector_path, disp_constants, height_column)


        


    




