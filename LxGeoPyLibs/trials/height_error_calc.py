from LxGeoPyLibs.dataset.vector_dataset import VectorDataset
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

def compute_height_error_sep(ref_vector_path, aligned_vector_path, ref_height_column, tar_height_column):
    
    saved_ids = set()
    ref_dst = VectorDataset(ref_vector_path, spatial_patch_size=(512,512), spatial_patch_overlap=0)
    tar_dst = VectorDataset(aligned_vector_path)
    ref_height = []
    error = []
    for c_window in tqdm.tqdm(ref_dst.patch_grid):

        ref_view = ref_dst._load_vector_features_window(c_window, ex_fields=[tar_height_column])
        ref_view.drop(ref_view[ref_view["id"].isin(saved_ids)].index, inplace=True)
        tar_view = tar_dst._load_vector_features_window(c_window, ex_fields=[ref_height_column])
        if ref_view.empty or tar_view.empty:
            continue
        
        ref_view["area"] = ref_view.area
        tar_view["area"] = tar_view.area

        inter_gdf = gpd.overlay(ref_view, tar_view)
        if inter_gdf.empty: continue
        inter_gdf["IOU"] = inter_gdf.area / (inter_gdf["area_1"]+inter_gdf["area_2"]-inter_gdf.area)

        err_df=inter_gdf.groupby("id_1").apply(lambda rows: (abs(rows[ref_height_column]-rows[tar_height_column])*rows["IOU"]).sum()/rows["IOU"].sum() )
        ref_view=ref_view.join(err_df.rename("err"), on="id").dropna()

        error.extend(ref_view.err.values.tolist())
        ref_height.extend(ref_view[ref_height_column].values.tolist())
        
        saved_ids.update(ref_view.id.values)

    df=pd.DataFrame({"ref_height":ref_height, "error":error})
    print("overall: " + str(df.error.mean()))
    print("[0,10]: " + str(df[df.ref_height<=10].error.mean()))
    print("[10,20]: " + str(df[(df.ref_height>10) & (df.ref_height<=20)].error.mean()))
    print("[20,50]: " + str(df[(df.ref_height>20) & (df.ref_height<=50)].error.mean()))
    print("[50,inf]: " + str(df[(df.ref_height>50)].error.mean()))
    pass

if __name__ == "__main__":
    
    in_vector1 = "/mnt/disk3/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/paris_tristereo_dhm_plus_external/gt_buildings/Buildings.shp"
    h_col_name1 = "AGL"
    in_vector2 = "/mnt/disk3/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/paris_tristereo_dhm_plus_external/alignment2/approach3/PHR1A_acq20180326_del736ec042/aligned.shp"
    h_col_name2 = "al_height"
    height_bins = [10, 20, 50]
    compute_height_error_sep(in_vector1, in_vector2, h_col_name1, h_col_name2)
    pass
    