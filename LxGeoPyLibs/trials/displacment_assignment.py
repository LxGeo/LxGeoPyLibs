
from LxGeoPyLibs.dataset.specific_datasets.multi_vector_dataset import MultiVectorDataset
import geopandas as gpd
import warnings;   warnings.filterwarnings("ignore")
import pandas as pd
import tqdm
import numpy as np

from LxGeoPyLibs.satellites.imd import IMetaData
from LxGeoPyLibs.satellites import formulas
from math import radians

################### used to assign height after alignment using lxFumble where transformed geometries don't have the same id also not exactly have the same area

def calc_disp(map1, map2, out_map, fl_tuple):
    """
    """
    vector_maps_dict = {"in_vector1" : map1,"in_vector2" : map2}

    duplicate_preprocessing = lambda gdf: gdf.drop_duplicates(subset=['id'])

    multi_dataset = MultiVectorDataset(vector_maps_dict, spatial_patch_size=(512,512),spatial_patch_overlap=10, in_fields=[], preprocessings=duplicate_preprocessing)

    out_crs = multi_dataset.sub_datasets_dict["in_vector1"].crs()
    
    saved_ids = set()
    mode="w"
    for views_dict in tqdm.tqdm(multi_dataset):
        
        gdf1 = views_dict["in_vector1"]
        gdf2 = views_dict["in_vector2"]
        if gdf1.empty or gdf2.empty:
            continue

        gdf1.drop(gdf1[gdf1["id"].isin(saved_ids)].index, inplace=True)
        gdf1["area"] = round(gdf1.area, 4)
        gdf1["centroid"] = gdf1.centroid
        gdf2["area"] = round(gdf2.area, 4)
        gdf2["centroid"] = gdf2.centroid

        inner_df = gdf1.join(gdf2.set_index('area'), on='area', lsuffix="", rsuffix="_right", how="inner")
        def centroids_to_disp(row):
            r_c = row["centroid_right"]
            l_c = row["centroid"]
            row["dx"], row["dy"] = r_c.x - l_c.x, r_c.y - l_c.y
            """if fl_tuple[0]>fl_tuple[1]:
                row["p_height"] = row["dx"] / fl_tuple[0]
            else:
                row["p_height"] = row["dy"] / fl_tuple[1]"""
            row["al_height"] = (abs(fl_tuple[0])*row["dx"]/fl_tuple[0] + abs(fl_tuple[1])*row["dy"]/fl_tuple[1]) / (abs(fl_tuple[0])+abs(fl_tuple[1]))
            return row
        inner_df = inner_df.apply(centroids_to_disp, axis=1)
        if inner_df.empty: continue
        
        to_save = gpd.GeoDataFrame(inner_df[["geometry", "dx", "dy", "al_height"]], crs=out_crs)
        saved_ids.update(inner_df["id"].values)
        if not to_save.empty:
            to_save.to_file(out_map, mode=mode)
            mode="a"
    

if __name__ == "__main__":
    
    in_vector1 = "/mnt/disk3/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/paris_tristereo_dhm_plus_external/prediction/PHR1A_acq20180326_del736ec042/fusion/buildings.shp"
    in_vector2 = "/mnt/disk3/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/paris_tristereo_dhm_plus_external/alignment2/lxFumble/PHR1A_acq20180326_del736ec042/"
    out_map = "/mnt/disk3/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/paris_tristereo_dhm_plus_external/alignment2/lxFumble/PHR1A_acq20180326_del736ec042/ref_height/buildings.shp"
    imd1 = "/mnt/disk3/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/paris_tristereo_dhm_plus_external/extracted/PHR1A_acq20180326_del736ec042/geo_PHR1A_acq20180326_del736ec042.imd"
    imd2 = "/mnt/disk3/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/paris_tristereo_dhm_plus_external/extracted/PHR1A_acq20180326_del1b382d54/geo_PHR1A_acq20180326_del1b382d54.imd"
    
    imd1 = IMetaData(imd1)
    imd2 = IMetaData(imd2)

    x_flow, y_flow = formulas.compute_roof2roof_constants(radians(imd1.satAzimuth()), radians(imd1.satElevation()),
     radians(imd2.satAzimuth()), radians(imd2.satElevation())
     )

    average_sims = calc_disp(in_vector1, in_vector2, out_map, (x_flow, y_flow))
    print(average_sims)
