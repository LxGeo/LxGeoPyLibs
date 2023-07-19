import os
import numpy as np
import geopandas as gpd
from LxGeoPyLibs.geometry.utils_rio import extents_to_profile
from LxGeoPyLibs.geometry.rasterizers import rasterize_from_profile
import seaborn as sns
from matplotlib import pyplot
import pandas as pd

get_dst_name_from_path = lambda path: os.path.basename(os.path.dirname(path)).split("_A_")[0].replace("_", " ")

def get_gdf_pixel_disps(input_gdf, resolution=0.3):
    """
    """
    flow_profile = extents_to_profile(input_gdf.total_bounds, crs=input_gdf.crs, count=2, dtype=np.float32)
    dispx = rasterize_from_profile(input_gdf.geometry, flow_profile, input_gdf.disp_x.values / -resolution)
    dispy = rasterize_from_profile(input_gdf.geometry, flow_profile, input_gdf.disp_y.values/ resolution)
    return dispx, dispy


if __name__ == "__main__":
    
    ref_paths=[
        "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Brazil_Vila_Velha_A_Neo/h_Brazil_Vila_Velha_A_Neo.shp",
        "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Ethiopia_Addis_Ababa_1_A_Neo/h_Ethiopia_Addis_Ababa_1_A_kaw.shp",
        "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/India_Mumbai_A_Neo/h_India_Mumbai_A_Neo_.shp",
        "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Kuwait_Kuwait_City_A_Neo/h_Kuwait_A.shp",
        "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Pakistan_Rawalpindi_A_Neo/h_Pakistan_A.shp",
        "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Qatar_Doha_A_Neo/h_buildings.shp",
        "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Sweden_Stockholm_A_Neo/h_buildings.shp",
    ]

    all_dfs=[]
    bins = np.linspace(0, 200, 50)
    for c_path in ref_paths:
        c_dst_name = get_dst_name_from_path(c_path)
        disp_x, disp_y = get_gdf_pixel_disps(gpd.read_file(c_path))
        disp_x, disp_y = abs(disp_x).ravel(), abs(disp_y).ravel()
        #pyplot.hist(disp_x, bins, alpha=0.5, label=c_dst_name)
        c_df =pd.DataFrame({"abs_disp_x":disp_x, "abs_disp_y":disp_y, "City": c_dst_name})
        all_dfs.append(c_df)
    
    all_dfs = pd.concat(all_dfs)
    sns.kdeplot(all_dfs[all_dfs.abs_disp_x>0], x="abs_disp_x",
                multiple="dodge", hue="City", fill=True, common_norm=False, palette="crest",
                alpha=.5, linewidth=0,)
    pyplot.show()