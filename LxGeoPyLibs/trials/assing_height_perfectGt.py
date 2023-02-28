from LxGeoPyLibs.satellites.imd import IMetaData
from LxGeoPyLibs.satellites.formulas import compute_roof2roof_constants
from LxGeoPyLibs.dataset.vector_dataset import VectorDataset
from LxGeoPyLibs.dataset.w_vector_dataset import WVectorDataset, WriteMode
import tqdm
import typing
import math
import numpy as np
import geopandas as gpd


def assign_height_mutually(in_vector_path1, in_vector_path2, out_vector_path1, out_vector_path2, disp_constants: typing.Tuple[float, float], height_column):
    """
    """
    in_dst1 = VectorDataset(in_vector_path1, spatial_patch_size=(512,512), spatial_patch_overlap=40)
    in_dst2 = VectorDataset(in_vector_path2, spatial_patch_size=(512,512), spatial_patch_overlap=40)
    new_fields = {height_column:"float", "disp_x":"float", "disp_y":"float"}
    out_schema1=in_dst1.fio_dataset().schema; out_schema1["properties"] |= new_fields
    out_schema2=in_dst2.fio_dataset().schema; out_schema2["properties"] |= new_fields
    out_dst1 = WVectorDataset(out_vector_path1, schema=out_schema1,crs=in_dst1.crs(), mode=WriteMode.overwrite)
    out_dst2 = WVectorDataset(out_vector_path2, schema=out_schema2,crs=in_dst2.crs(), mode=WriteMode.overwrite)
    #transform_fn = partial(baseToRoof, dDx=disp_constants[0], dDy=disp_constants[1], height_column=height_column)

    for c_window in tqdm.tqdm(in_dst1.patch_grid, desc="Window processing "):
        in_view1 = in_dst1._load_vector_features_window(c_window, ex_fields=["id", "STATUS"])
        in_view1 = in_view1[in_view1.geometry.is_valid]
        in_view2 = in_dst2._load_vector_features_window(c_window, ex_fields=["id", "STATUS"])
        in_view2 = in_view2[in_view2.geometry.is_valid]
        #in_view1["height_column"] = in_view1.apply(lambda row: transform_fn(row), axis=1)

        df_join = in_view1.set_index('BUILD_ID').join(in_view2.set_index('BUILD_ID'), lsuffix="_l", rsuffix="_r")
        df_join["disp_x"] = df_join["geometry_l"].centroid.x - df_join["geometry_r"].centroid.x
        df_join["disp_y"] = df_join["geometry_l"].centroid.y - df_join["geometry_r"].centroid.y

        df_join[height_column] = np.sqrt((df_join["disp_x"]/disp_constants[0])**2 + (df_join["disp_y"]/disp_constants[1])**2)
        df_join = df_join.dropna(subset=[height_column]).reset_index()

        columns_to_save_names = lambda suffix: ["BUILD_ID", f"id{suffix}", f"geometry{suffix}", "disp_x", "disp_y", height_column]
        out_dst1.add_feature(
            gpd.GeoDataFrame(df_join[columns_to_save_names("_l")].rename({"id_l": "id", "geometry_l": "geometry"}, axis='columns'), crs=in_view1.crs,geometry="geometry")
            
            )
        out_dst2.add_feature(
            gpd.GeoDataFrame(df_join[columns_to_save_names("_r")].rename({"id_r": "id", "geometry_r": "geometry"}, axis='columns'),crs=in_view2.crs,geometry="geometry")
                            )
    return

if __name__ == "__main__":

    in_vector_path1 = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Brazil_Vila_Velha_A_Neo/Brazil_Vila_Velha_A_Neo.shp"
    in_imd_path1 = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Brazil_Vila_Velha_A_Neo/ade27182-11da-483d-8fef-fb1a76b00568.IMD"
    in_vector_path2 = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Brazil_Vila_Velha_B_Neo/Brazil_Vila_Velha_B_Neo.shp"
    in_imd_path2 = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Brazil_Vila_Velha_B_Neo/b25caa5a-190b-4fa9-957e-43816cc462c2.IMD"
    out_vector_path1 = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Brazil_Vila_Velha_A_Neo/h_Brazil_Vila_Velha_A_Neo.shp"
    out_vector_path2 = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Brazil_Vila_Velha_B_Neo/h_Brazil_Vila_Velha_B_Neo.shp"

    imd1 = IMetaData(in_imd_path1)
    imd2 = IMetaData(in_imd_path2)
    disp_constants = compute_roof2roof_constants(math.radians(imd1.satAzimuth()),
         math.radians(imd1.satElevation()),
         math.radians(imd2.satAzimuth()),
         math.radians(imd2.satElevation())
        )
    height_column = "al_height"
    assign_height_mutually(in_vector_path1, in_vector_path2, out_vector_path1, out_vector_path2, disp_constants, height_column)