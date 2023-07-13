import os
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np
import seaborn as sb
import momepy

def add_nearest_neighbour_distance(gdf):

    gdf['NN'] = None

    # Loop through the GeoDataFrame and compute the nearest neighbor distance for each building
    for idx, row in gdf.iterrows():
        # Create a new GeoDataFrame without the current building
        gdf_without_current = gdf.loc[gdf.index != idx]
        
        # Compute the distances from the current building to all other buildings
        distances = gdf_without_current.distance(row["geometry"])
        
        # Find the minimum distance and assign it to the 'nearest_neighbor' column
        gdf.loc[idx, 'NN'] = distances.min()

def add_tesselation_intensity(gdf):
    gdf["unique_id"] = gdf.index
    limit = momepy.buffered_limit(gdf, 20)
    tessellation = momepy.Tessellation(gdf, "unique_id", limit, verbose=False, segment=1)
    tessellation = tessellation.tessellation
    tessellation["area"] = tessellation.area
    AR = momepy.AreaRatio(tessellation, gdf, 'area', 'area', "unique_id").series
    gdf['TI'] = AR
    gdf=gdf.drop(["unique_id"], axis=1)


height_column_name = "al_height"
conf_column_name = "hrel"

cities_dfs = []

def get_city_df(c_file_path, city_name=None):

    gdf = gpd.read_file(c_file_path)
    gdf = gdf[gdf[conf_column_name]>0.5].reset_index()
    gdf["area"] = gdf.area
    
    add_tesselation_intensity(gdf)
    add_nearest_neighbour_distance(gdf)

    if not city_name:
        city_name = os.path.basename(os.path.dirname(c_file_path))
    c_df = pd.DataFrame(
            {
                'City': [city_name] * len(gdf),
                "NN": gdf["NN"],
                "TI": gdf["TI"],
                "Area": gdf.area,
                "Height": gdf[height_column_name]
            }
        )
    
    c_df = c_df[c_df["NN"] < c_df["NN"].quantile(0.95) ]
    return c_df


if __name__=="__main__":

    file_paths = [
        ("C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Brazil_Vila_Velha_A_Neo/h_Brazil_Vila_Velha_A_Neo.shp", "Brazil Vila Velha"),
        ("C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Ethiopia_Addis_Ababa_1_A_Neo/h_Ethiopia_Addis_Ababa_1_A_kaw.shp", "Ethiopia Addis Ababa"),
        ("C:/DATA_SANDBOX/Alignment_Project/PerfectGT/India_Mumbai_A_Neo/h_India_Mumbai_A_Neo_.shp", "India Mumbai"),
        ("C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Kuwait_Kuwait_City_A_Neo/h_Kuwait_A.shp", "Kuwait Kuwait"),
        ("C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Pakistan_Rawalpindi_A_Neo/h_Pakistan_A.shp", "Pakistan Rawalpindi"),
        ("C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Qatar_Doha_A_Neo/h_buildings.shp", "Qatar Doha"),
        ("C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Sweden_Stockholm_A_Neo/h_buildings.shp", "Sweden Stockholm"),
    ]

    boxes_colors=["beige", "lightblue", "lightpink", "lightcyan", "anqtiquewhite", "lightsteelblue"]

    grouped_file_output = "C:/Users/geoimage/Music/grouped.csv"

    if (not os.path.isfile(grouped_file_output)):
        pool = mp.Pool(mp.cpu_count())
        cities_dfs = pool.starmap(get_city_df, file_paths)
        grouped_df = pd.concat(cities_dfs)
        grouped_df.to_csv(grouped_file_output, index=False)
    else:
        grouped_df = pd.read_csv(grouped_file_output, )

    NN_sorted_cities_names = list(grouped_df.groupby("City").agg({"NN": lambda vals: np.percentile(vals , 75) }).sort_values("NN", ascending=True).index.values)

    measures={"Area":"meters^2", "TI":"ratio" }

    column_to_plot = "TI"
    measure = measures.get(column_to_plot, "meters")
    sb.boxplot( x = 'City',y = column_to_plot, data = grouped_df, order= NN_sorted_cities_names).set(ylabel=f"{column_to_plot} ({measure})")
    sb.boxplot( y = 'City',x = column_to_plot, data = grouped_df, order= NN_sorted_cities_names, orient="h").set(xlabel=f"{column_to_plot} ({measure})")
    sb.kdeplot(data=grouped_df.reset_index(), x="TI", hue="City", fill=True, common_norm=False, cut=0, alpha=0.5, hue_order=NN_sorted_cities_names)
    """bplot = grouped_df.boxplot(by='City', column=["NN"], patch_artist=True, return_type='dict')
    for c_box, c_color in zip( bplot["NN"]["boxes"], boxes_colors):
        c_box.set(facecolor=c_color)
    """
    plt.show()
    pass
