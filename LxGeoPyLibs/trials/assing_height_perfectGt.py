from LxGeoPyLibs.satellites.imd import IMetaData
from LxGeoPyLibs.satellites.formulas import compute_roof2roof_constants
from LxGeoPyLibs.dataset.vector_dataset import VectorDataset
from LxGeoPyLibs.dataset.w_vector_dataset import WVectorDataset, WriteMode
from LxGeoPyLibs.projection_utils import is_utm, utm_zone_code, reproject_geom
from LxGeoPyLibs import _logger
import tqdm
import typing
import math
import numpy as np
import geopandas as gpd
from shapely import affinity
from shapely.geometry import Point, box
from pyproj import CRS
import click


def assign_height_mutually(in_vector_path1, in_vector_path2, out_vector_path1, out_vector_path2, disp_constants: typing.Tuple[float, float], height_column):
    """
    """
    in_dst1 = VectorDataset(in_vector_path1, spatial_patch_size=(10000,10000), spatial_patch_overlap=40)
    in_dst2 = VectorDataset(in_vector_path2, spatial_patch_size=(10000,10000), spatial_patch_overlap=40)

    if (in_dst1.crs()!=in_dst2.crs()):
        _logger.warn("Vector datasets have different CRS!")
        return

    requiers_reprojection = not is_utm(in_dst1.crs())
    if requiers_reprojection:
        bounds_box = box(*in_dst1.bounds())
        zone_centroid = bounds_box.centroid
        geo_zone_centroid = reproject_geom(zone_centroid, in_dst1.crs(), CRS("EPSG:4326"))
        respective_utm_crs = CRS("epsg:{}".format( utm_zone_code(geo_zone_centroid.y, geo_zone_centroid.x) ))

    new_fields = {height_column:"float", "disp_x":"float", "disp_y":"float"}
    out_schema1=in_dst1.fio_dataset().schema; out_schema1["properties"] |= new_fields
    out_schema2=in_dst2.fio_dataset().schema; out_schema2["properties"] |= new_fields
    out_dst1 = WVectorDataset(out_vector_path1, schema=out_schema1,crs=in_dst1.crs(), mode=WriteMode.overwrite)
    out_dst2 = WVectorDataset(out_vector_path2, schema=out_schema2,crs=in_dst2.crs(), mode=WriteMode.overwrite)

    for c_window in tqdm.tqdm(in_dst1.patch_grid, desc="Window processing "):
        
        in_view1 = in_dst1._load_vector_features_window(c_window, ex_fields=["STATUS"], field_names_rename_map={"id":"bid"})
        in_view1 = in_view1[in_view1.geometry.is_valid]
        in_view2 = in_dst2._load_vector_features_window(c_window, ex_fields=["STATUS"], field_names_rename_map={"id":"bid"})
        in_view2 = in_view2[in_view2.geometry.is_valid]
        #in_view1["height_column"] = in_view1.apply(lambda row: transform_fn(row), axis=1)
        if(in_view1.empty and in_view2.empty):
            continue
        
        df_join = in_view1.set_index('bid').join(in_view2.set_index('bid'), lsuffix="_l", rsuffix="_r")
        df_join = df_join[df_join.index.notnull()]
        df_join = df_join.reset_index()
        df_join["match_distance"] = df_join[~((df_join["geometry_l"]==None) | (df_join["geometry_r"]==None))].apply(lambda row: row["geometry_l"].distance(row["geometry_r"]), axis=1)
        df_join = df_join.sort_values("match_distance")
        df_join = df_join[~df_join["bid"].duplicated(keep='first')].set_index("bid")

        if( not requiers_reprojection ):
            reprojectd_left_gdf = gpd.GeoDataFrame(geometry=df_join["geometry_l"], crs=in_dst1.crs())
            reprojectd_right_gdf = gpd.GeoDataFrame(geometry=df_join["geometry_r"], crs=in_dst2.crs())
        else:
            reprojectd_left_gdf = gpd.GeoDataFrame(geometry=df_join["geometry_l"], crs=in_dst1.crs()).to_crs(respective_utm_crs)
            reprojectd_right_gdf = gpd.GeoDataFrame(geometry=df_join["geometry_r"], crs=in_dst2.crs()).to_crs(respective_utm_crs)
        
        reprojectd_left_gdf = reprojectd_left_gdf[reprojectd_left_gdf.geometry.is_valid]
        reprojectd_right_gdf = reprojectd_right_gdf[reprojectd_right_gdf.geometry.is_valid]

        disp_x = reprojectd_left_gdf.centroid.x - reprojectd_right_gdf.centroid.x
        disp_y = reprojectd_left_gdf.centroid.y - reprojectd_right_gdf.centroid.y
        df_join["disp_x"] = disp_x[~disp_x.index.duplicated(keep='first')]
        df_join["disp_y"] = disp_y[~disp_y.index.duplicated(keep='first')]
        

        if (abs(disp_constants[1])>abs(disp_constants[0])):
            df_join[height_column] = abs(df_join["disp_y"]/disp_constants[1])
        else:
            df_join[height_column] = abs(df_join["disp_x"]/disp_constants[0])

        df_join = df_join.dropna(subset=[height_column])
        reprojectd_right_gdf = reprojectd_right_gdf.loc[df_join.index]
        df_join=df_join.reset_index()
        ### computing confidence values
        # Translate left to right
        aligned_geometry = reprojectd_right_gdf.apply(lambda row: affinity.translate(row["geometry"], df_join.loc[df_join["bid"]==row.name, "disp_x"].values[0], df_join.loc[df_join["bid"]==row.name, "disp_y"].values[0]), axis=1)
        intersection_geometry = aligned_geometry.intersection(reprojectd_left_gdf)
        union_geometry = aligned_geometry.union(reprojectd_left_gdf)
        
        df_join = df_join.set_index("bid")
        confidence_column_name = "hrel"
        df_join[confidence_column_name] = intersection_geometry.area / union_geometry.area

        df_join = df_join.reset_index()

        columns_to_save_names = lambda suffix: ["bid", f"id{suffix}", f"geometry{suffix}", "disp_x", "disp_y", height_column, confidence_column_name]
        dst1_features = gpd.GeoDataFrame(df_join[columns_to_save_names("_l")].rename({"id_l": "id", "geometry_l": "geometry"}, axis='columns'), crs=in_view1.crs,geometry="geometry")
        dst1_features["disp_x"]*=-1; dst1_features["disp_y"]*=-1;
        out_dst1.add_feature(dst1_features)
        dst2_features=gpd.GeoDataFrame(df_join[columns_to_save_names("_r")].rename({"id_r": "id", "geometry_r": "geometry"}, axis='columns'),crs=in_view2.crs,geometry="geometry")
        out_dst2.add_feature(dst2_features)
    
    return

def run(in_v1, in_imd1, in_v2, in_imd2, out_v1, out_v2):
    imd1 = IMetaData(in_imd1)
    imd2 = IMetaData(in_imd2)
    disp_constants = compute_roof2roof_constants(math.radians(imd1.satAzimuth()),
         math.radians(imd1.satElevation()),
         math.radians(imd2.satAzimuth()),
         math.radians(imd2.satElevation())
        )
    height_column = "al_height"
    assign_height_mutually(in_v1, in_v2, out_v1, out_v2, disp_constants, height_column)

@click.command()
@click.option('-in_v1', '--input_vector_1', required=True, type=click.Path(exists=True), help="First vector file path")
@click.option('-iimd1', '--input_imd_1', type=click.Path(exists=True), help="First reference image imd file path")
@click.option('-in_v2', '--input_vector_2', required=True, type=click.Path(exists=True), help="Second vector file path")
@click.option('-iimd2', '--input_imd_2', type=click.Path(exists=True), help="Second reference image imd file path")
@click.option('-out_v1', '--output_vector_1', type=click.Path(), help="First output vector file path")
@click.option('-out_v2', '--output_vector_2', type=click.Path(), help="Second output vector file path")
def main_cmd(input_vector_1, input_imd_1, input_vector_2, input_imd_2, output_vector_1, output_vector_2):
    run(input_vector_1, input_imd_1, input_vector_2, input_imd_2, output_vector_1, output_vector_2)

if __name__ == "__main__":
    main_cmd()

if __name__ == "__test__":
    in_vector_path1 = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Sweden_Stockholm_A_Neo/buildings.shp"
    in_imd_path1 = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Sweden_Stockholm_A_Neo/8ec751d4-892d-49de-8f9e-099ea9f1228c.IMD"

    in_vector_path2 = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Sweden_Stockholm_B_Neo/buildings.shp"
    in_imd_path2 = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Sweden_Stockholm_B_Neo/e62acdd2-a406-41d8-ba3f-57254d7fb6cd.IMD"

    out_vector_path1 = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Sweden_Stockholm_A_Neo/h_buildings.shp"
    out_vector_path2 = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Sweden_Stockholm_B_Neo/h_buildings.shp"
    run(in_vector_path1, in_imd_path1, in_vector_path2, in_imd_path2, out_vector_path1, out_vector_path2)

    