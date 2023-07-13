

import geopandas as gpd
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os


def get_dataset_error(shp1_path, shp2_path):
    gdf1=gpd.read_file(shp1_path)
    gdf2=gpd.read_file(shp2_path)
    dataset_name = os.path.basename(os.path.dirname(shp1_path)).split("_A_")[0]
    dataset_name=dataset_name.replace("_", " ")

    gdf1['Height class'] = pd.cut(gdf1['al_height'],
                            bins=[0, 10, 20, 40, float('Inf')],
                            labels=['0-10', '10-20', '20-40', '40-inf'])

    jointure = gdf1.set_index("bid").join(gdf2.set_index("id"), lsuffix="l_")
    jointure["Dataset name"]=dataset_name
    

    jointure["err_disp_x"] = abs( jointure["disp_x"] - (jointure["DISP_X"]))
    jointure["err_disp_y"] = abs( jointure["disp_y"] - (jointure["DISP_Y"]))

    jointure["Displacment error"] = np.sqrt(jointure["err_disp_x"]**2+jointure["err_disp_y"]**2)
    q=0.95
    inliers = jointure[ ((jointure["err_disp_x"]< jointure["err_disp_x"].quantile(q)) & (jointure["err_disp_y"]< jointure["err_disp_y"].quantile(q))) ]
    #inliers = inliers[inliers["confidence"]<0.15]
    return inliers

def get_all_datasets_error():

    ref_paths=[
        "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Brazil_Vila_Velha_A_Neo/h_Brazil_Vila_Velha_A_Neo.shp",
        "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Ethiopia_Addis_Ababa_1_A_Neo/h_Ethiopia_Addis_Ababa_1_A_kaw.shp",
        "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/India_Mumbai_A_Neo/h_India_Mumbai_A_Neo_.shp",
        "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Kuwait_Kuwait_City_A_Neo/h_Kuwait_A.shp",
        "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Pakistan_Rawalpindi_A_Neo/h_Pakistan_A.shp",
        "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Qatar_Doha_A_Neo/h_buildings.shp",
        "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Sweden_Stockholm_A_Neo/h_buildings.shp",
    ]

    sub_paths=[
        "C:/DATA_SANDBOX/Alignment_Project/alignment_results/lxProximityAlign2D/Brazil_Vila_Velha_gt_v2tov1/aligned.shp",
        "C:/DATA_SANDBOX/Alignment_Project/alignment_results/lxProximityAlign2D/Ethiopia_Addis_Ababa_gt_v2tov1/aligned.shp",
        "C:/DATA_SANDBOX/Alignment_Project/alignment_results/lxProximityAlign2D/India_Mumbai_gt_v2tov1/aligned.shp",
        "C:/DATA_SANDBOX/Alignment_Project/alignment_results/lxProximityAlign2D/Kuwait_Kuwait_gt_v2tov1/aligned.shp",
        "C:/DATA_SANDBOX/Alignment_Project/alignment_results/lxProximityAlign2D/Pakistan_Rawalpindi_gt_v2tov1/aligned.shp",
        "C:/DATA_SANDBOX/Alignment_Project/alignment_results/lxProximityAlign2D/Qatar_Doha_gt_v2tov1/aligned.shp",
        "C:/DATA_SANDBOX/Alignment_Project/alignment_results/lxProximityAlign2D/Sweden_Stockholm_gt_v2tov1/aligned.shp",
    ]

    datasets_error_dfs=[]
    for (ref,sub) in zip(ref_paths, sub_paths):
        datasets_error_dfs.append(get_dataset_error(ref,sub)[["Displacment error", "Height class", "Dataset name"]])
    
    combined_error_df = pd.concat(datasets_error_dfs)
    return combined_error_df

if __name__=="__main__":

    # shp1 is height assigned
    #shp1_path="C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Brazil_Vila_Velha_A_Neo/h_Brazil_Vila_Velha_A_Neo.shp"
    # shp2 is aligned shapefile
    #shp2_path="C:/DATA_SANDBOX/Alignment_Project/alignment_results/lxProximityAlign2D/Brazil_Vila_Velha_gt_v2tov1/aligned.shp"
    #inliers = get_dataset_error(shp1_path, shp2_path)
    #sns.violinplot(inliers, x="Displacment error", y='Height class')
    #plt.show()
    combined_error_df=get_all_datasets_error()
    print(f"count of geometries: {len(combined_error_df)}")

    #sns.violinplot(combined_error_df, x="Displacment error", y="Dataset name", hue='Height class', orient="h")
    sns.boxplot(combined_error_df, y="Displacment error", x="Dataset name", hue='Height class', orient="v")
    #plt.grid(linewidth = 1.5, alpha = 0.25)
    plt.grid(which = "major", linewidth = 1)
    plt.grid(which = "minor", linewidth = 0.2)
    plt.minorticks_on()
    plt.show()
    
    a=0
