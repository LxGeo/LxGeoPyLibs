

import os
from enum import Enum
import fiona

class WriteMode(Enum):
    new = 0
    overwrite = 1
    append = 2

def get_layer_feature_ids(vector_path):
    """
    Returns all features ids in a layer as a set
    """
    ids_set = set()
    for c_feature in fiona.open(vector_path):        
        ids_set.update(c_feature["id"])
    return ids_set

class WVectorDataset():

    def __init__(self, vector_path:str, mode=WriteMode.new, driver=None, schema=None, crs=None):

        assert (mode != WriteMode.new) or not os.path.exists(vector_path), "WVectorDataset with mode 'new' for existing path !"
        self.vector_path = vector_path

        if (mode == WriteMode.append):
            open_mode = "a"
        else:
            open_mode = "w"
            driver = "ESRI Shapefile"
        
        self.dataset = fiona.open(vector_path, open_mode, driver=driver, schema=schema, crs=crs)

        if mode in (WriteMode.new, WriteMode.overwrite):
            self.saved_ids = set()
            self.c_write_mode="w"
        else:
            self.c_write_mode="a"
            self.saved_ids = get_layer_feature_ids(self.vector_path)
    

    def get_layer_feature_ids(self):
        """
        Returns all features ids in a layer as a set
        """
        ids_set = set()
        for c_feature in self.dataset:        
            ids_set.update(c_feature["id"])
        return ids_set
    
    def add_feature(self, features_gdf):
        """
        """
        features_gdf.drop(features_gdf[features_gdf["id"].isin(self.saved_ids)].index, inplace=True)
        if not features_gdf.empty:
            self.saved_ids.update(features_gdf["id"].values)
            features_gdf.to_file(self.vector_path, mode=self.c_write_mode)
            self.c_write_mode="a"
