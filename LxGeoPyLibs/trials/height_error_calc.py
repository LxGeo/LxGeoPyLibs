from LxGeoPyLibs.dataset.vector_dataset import VectorDataset
from LxGeoPyLibs.dataset.w_vector_dataset import WVectorDataset, WriteMode
import typing
import tqdm
import geopandas as gpd
from libpysal.weights import Rook
import pygeos
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


############## Used to compute height error between two vectors spatially aligned. Per block or per seperatly (per instance)

def get_geoms_window(geom_iter):
    all_coords=[]
    for geom in geom_iter:
        geom=geom.exterior
        all_coords.extend(geom.coords)
    min_x = min(all_coords, key=lambda x: x[0])[0]
    min_y = min(all_coords, key=lambda x: x[1])[1]
    max_x = max(all_coords, key=lambda x: x[0])[0]
    max_y = max(all_coords, key=lambda x: x[1])[1]
    return pygeos.box(min_x, min_y, max_x, max_y)

def compute_agg_error(ref_dataset, aligned_gdf):

    ref_view = ref_dataset._load_vector_features_window(aligned_gdf.total_bounds)
    inter_gdf = gpd.overlay(ref_view, aligned_gdf)
    if inter_gdf.empty:
        return None, None
    c_ref_comp_height = (inter_gdf["AGL"]*inter_gdf.area).sum()/inter_gdf.area.sum()
    tar_height = inter_gdf["al_height"][0]
    #c_comp_error = abs(c_ref_comp_height - tar_height)
    return c_ref_comp_height, tar_height

def compute_height_error(ref_vector_path, aligned_vector_path, ref_height_column, tar_height_column, height_bins):
    """
    """

    al_gdf= gpd.read_file(aligned_vector_path)
    sp_weights = Rook.from_dataframe(al_gdf)
    al_gdf["comp_id"]=sp_weights.component_labels
    #al_comp_windows = al_gdf.groupby("comp_id").agg({"geometry": get_geoms_window})

    ref_dst = VectorDataset(ref_vector_path)
    ref_height = []
    tar_height = []
    tar_area = []

    error_tasks = []
    for c_comp_idx in tqdm.tqdm(range(sp_weights.n_components)):
        aligned_gdf_view = al_gdf[al_gdf["comp_id"]==c_comp_idx]
        tar_area.append(aligned_gdf_view.area.sum())
        error_tasks.append( compute_agg_error(ref_dst, aligned_gdf_view))

    for c_task in error_tasks:
        output = c_task#.result()
        ref_height.append(output[0])   
        tar_height.append(output[1])   

    df=pd.DataFrame({"ref_height":ref_height, "tar_height":tar_height, "tar_area":tar_area})
    df["error"] = abs(df["ref_height"]-df["tar_height"])
    df["weight"] = df.tar_area/df.tar_area.max()
    df["w_err"] = df["error"] * df["weight"]

    calc_err = lambda v: v.w_err.sum()/v.weight.sum()

    print("overall: " + str(calc_err(df)))
    print("[0,10]: " +str(calc_err(df[df.ref_height<=10])))
    print("[10,20]: " +str(calc_err(df[(df.ref_height>10) & (df.ref_height<=20)])))
    print("[20,50]: " +str(calc_err(df[(df.ref_height>20) & (df.ref_height<=50)])))
    print("[50,inf]: " +str(calc_err(df[(df.ref_height>50)])))
    
    a=0
    pass

    """for c_comp_idx, c_window in tqdm.tqdm(enumerate(al_comp_windows.geometry), desc="Window processing "):
        ref_view = ref_dst._load_vector_features_window(c_window)
        al_view = al_gdf[al_gdf["comp_id"]==c_comp_idx]        
        inter_gdf = gpd.overlay(ref_view, al_view)
        c_ref_comp_height = (inter_gdf["AGL"]*inter_gdf.area).sum()/inter_gdf.area.sum()
        c_comp_error = abs(c_ref_comp_height - inter_gdf["al_height"][0])
        ref_height.append( c_ref_comp_height )
        height_error.append( c_comp_error )
    return"""

def compute_height_error_sep(ref_vector_path, aligned_vector_path, out_vector, ref_height_column, tar_height_column, height_bins):
    
    err_column_name = "h_err"
    saved_ids = set()
    ref_dst = VectorDataset(ref_vector_path, spatial_patch_size=(512,512), spatial_patch_overlap=0)
    tar_dst = VectorDataset(aligned_vector_path)
    out_dst = WVectorDataset(out_vector, crs=ref_dst.crs(), mode=WriteMode.overwrite)

    ref_height = []
    error = []
    for c_window in tqdm.tqdm(ref_dst.patch_grid):

        ref_view = ref_dst._load_vector_features_window(c_window, ex_fields=[tar_height_column, "id"])
        ref_view.drop(ref_view[ref_view["id"].isin(saved_ids)].index, inplace=True)
        tar_view = tar_dst._load_vector_features_window(c_window, ex_fields=[ref_height_column, "id"])
        if ref_view.empty or tar_view.empty:
            continue
        
        ref_view["area"] = ref_view.area
        tar_view["area"] = tar_view.area

        inter_gdf = gpd.overlay(ref_view, tar_view)
        if inter_gdf.empty: continue
        inter_gdf["IOU"] = inter_gdf.area / (inter_gdf["area_1"]+inter_gdf["area_2"]-inter_gdf.area)

        err_df=inter_gdf.groupby("id_1").apply(lambda rows: (abs(rows[ref_height_column]-rows[tar_height_column])*rows["IOU"]).sum()/rows["IOU"].sum() )
        ref_view=ref_view.join(err_df.rename(err_column_name), on="id")#.dropna()
        ref_view.loc[ref_view[ref_height_column].isna(), err_column_name] = None # keep nan assigned heights to nan

        out_dst.add_feature(ref_view.drop(["area"], axis=1))

        error.extend(ref_view[err_column_name].values.tolist())
        ref_height.extend(ref_view[ref_height_column].values.tolist())
        
        saved_ids.update(ref_view.id.values)

    df=pd.DataFrame({"ref_height":ref_height, "error":error})
    print("overall: " + str(df.error.mean()))
    for c_bin_idx in range(len(height_bins)-1):
        min_val = height_bins[c_bin_idx]
        max_val = height_bins[c_bin_idx+1]
        sub_df = df[(df.ref_height>min_val) & (df.ref_height<=max_val)]
        print(f"[{min_val},{max_val}]: count({len(sub_df)}): " + str(sub_df.error.mean()))
    pass

if __name__ == "__main__":
    
    out_vector = "C:/DATA_SANDBOX/Alignment_Project/sgbm_pipeline/perfect_gt_brazil/height_assesment_DL/Brazil_Vila_Velha_B_GT_map_error.shp"
    in_vector1 = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Brazil_Vila_Velha_B_Neo/preds/h_build_poly.shp"
    h_col_name1 = "h_mean"
    in_vector2 = "C:/DATA_SANDBOX/Alignment_Project/sgbm_pipeline/perfect_gt_brazil/height_assesment/Brazil_Vila_Velha_B_GT_map_error.shp"
    h_col_name2 = "al_height"
    height_bins = [0, 5, 10, 20, 50, 100]
    compute_height_error_sep(in_vector1, in_vector2, out_vector, h_col_name1, h_col_name2, height_bins)
    pass
    