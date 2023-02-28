
from LxGeoPyLibs.dataset.specific_datasets.multi_vector_dataset import MultiVectorDataset
import geopandas as gpd
import warnings;   warnings.filterwarnings("ignore")
import pandas as pd
import tqdm
import numpy as np

from LxGeoPyLibs.geometry.metrics.distances import polygon_distances

def map_similarities(map1, map2, distances_set):
    """
    """
    vector_maps_dict = {"in_vector1" : map1,"in_vector2" : map2}

    duplicate_preprocessing = lambda gdf: gdf.drop_duplicates(subset=['id'])

    multi_dataset = MultiVectorDataset(vector_maps_dict, spatial_patch_size=(512,512),spatial_patch_overlap=0, in_fields=[], preprocessings=duplicate_preprocessing)

    out_gdf = gpd.GeoDataFrame(columns=["dx", "dy", "geometry"], crs=multi_dataset.sub_datasets_dict["in_vector1"].crs())
    
    saved_ids = set()
    overall_similarities={ "similarity_"+d.name:[] for d in distances_set}

    for views_dict in tqdm.tqdm(multi_dataset):
        
        gdf1 = views_dict["in_vector1"]
        gdf2 = views_dict["in_vector2"]
        if gdf1.empty or gdf2.empty:
            continue
        gdf1.drop(gdf1[gdf1["id"].isin(saved_ids)].index, inplace=True)
        gdf1["area"] = gdf1.area
        gdf1["geom"] = gdf1.geometry
        gdf2["area"] = gdf2.area
        gdf2["geom"] = gdf2.geometry

        inter_gdf = gpd.overlay(gdf1, gdf2)
        inter_gdf["IOU"] = inter_gdf.area / (inter_gdf["area_1"]+inter_gdf["area_2"]-inter_gdf.area)

        inter_gdf.drop(inter_gdf[inter_gdf["IOU"]<0.05].index, inplace=True)
        if (len(inter_gdf)==0):
            continue
        
        for c_distance in distances_set:
            inter_gdf[c_distance.name] = c_distance(inter_gdf["geom_1"], inter_gdf["geom_2"])
            inter_gdf["similarity_"+c_distance.name] = inter_gdf["IOU"] / np.log(inter_gdf[c_distance.name] + np.e)


        #inter_gdf["chamfer_distance"] = polygon_distances.chamfer_distance( inter_gdf["geom_1"], inter_gdf["geom_2"] )
        #inter_gdf["hausdorff_distance"] = polygon_distances.hausdorff_distance( inter_gdf["geom_1"], inter_gdf["geom_2"] )
        #inter_gdf["polis_distance"] = polygon_distances.polis_distance( inter_gdf["geom_1"], inter_gdf["geom_2"] )

        #inter_gdf["similarity_chamfer"] = inter_gdf["IOU"] / np.log(inter_gdf["chamfer_distance"] + np.e)
        #inter_gdf["similarity_hausdorff"] = inter_gdf["IOU"] / np.log(inter_gdf["hausdorff_distance"] + np.e)
        #inter_gdf["similarity_polis"] = inter_gdf["IOU"] / np.log(inter_gdf["polis_distance"] + np.e)
        
        aggregator_def = { c_sim: "mean" for c_sim in overall_similarities }

        #similarity_serie = inter_gdf.groupby(["id_1"]).agg({"similarity_chamfer":"mean", "similarity_hausdorff":"mean", "similarity_polis":"mean"})
        similarity_serie = inter_gdf.groupby(["id_1"]).agg(aggregator_def)

        for k_sim in overall_similarities:
            overall_similarities[k_sim].extend(similarity_serie[k_sim].to_list())
        
        saved_ids.update(similarity_serie.index.values)
        pass
    
    average_sims = { k: np.average(v) for k,v in overall_similarities.items() }
    return average_sims

if __name__ == "__main__":
    import json
    
    in_vector1 = "/home/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/BRAGANCA/DL/ortho1/rooftop/build-poly.shp"
    in_vector2 = "/home/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/BRAGANCA/alignment/lxfumble/bing/"

    distances_set = set([polygon_distances.chamfer_distance, polygon_distances.hausdorff_distance, polygon_distances.polis_distance])
    average_sims = map_similarities(in_vector1, in_vector2, distances_set)
    print(average_sims)

if __name__== "__main2__":
    datasets_eval_paths= [
            "/home/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/Brazil/belford_east/alignment/eval.txt",
            "/home/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/BRAGANCA/alignment/eval.txt",
            "/home/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/Funchal/alignment/eval.txt",
            "/home/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/paris_tristereo_dhm_plus_external/alignment/eval.txt",
        ]
    datasets_keys = [
        "Belford (BR)",
        "Braganca (PT)",
        "Funchal (PT)",
        "Paris (FR)"
    ]

    datasets_dfs = []
    for c_dataset_eval_path in datasets_eval_paths:    
        with open(c_dataset_eval_path) as file:
            js= json.loads(file.read().replace("'", '"'))    
        c_target_dfs=[]
        c_target_keys=[]
        for c_target_key, c_target in js.items():
            al = c_target["algined_sims"]
            al["ref"] = c_target["misaligned_sims"]
            c_df=pd.DataFrame.from_dict(al, "index")
            c_df = c_df / c_df.loc["ref"]
            c_df = c_df.drop("ref")
            c_target_dfs.append(c_df)
            c_target_keys.append(c_target_key)    
        c_dataset_df = pd.concat([df.T.stack() for df in c_target_dfs], 1, keys=c_target_keys).T
        datasets_dfs.append(c_dataset_df)


    tf=pd.concat(datasets_dfs,keys=datasets_keys)




