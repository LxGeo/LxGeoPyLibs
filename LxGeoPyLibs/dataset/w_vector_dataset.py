

import os
from enum import Enum
import fiona

class WriteMode(Enum):
    new = 0
    overwrite = 1
    append = 2


class WVectorDataset(object):

    def __init__(self, vector_path:str, mode=WriteMode.new, driver="ESRI Shapefile", schema=None, crs=None):

        assert (mode != WriteMode.new) or not os.path.exists(vector_path), "WVectorDataset with mode 'new' for existing path !"
        self.vector_path = vector_path
        self.crs=crs
        
        self.saved_ids = set()

        if mode in (WriteMode.new, WriteMode.overwrite):
            self.c_write_mode="w"
        else:
            self.c_write_mode="a"
            with fiona.open(vector_path) as dataset:                
                for c_feature in dataset:        
                    self.saved_ids.update(c_feature["id"])
        
    def add_feature(self, features_gdf):
        """
        """
        features_gdf.drop(features_gdf[features_gdf["id"].isin(self.saved_ids)].index, inplace=True)
        if not features_gdf.empty:
            self.saved_ids.update(features_gdf["id"].values)
            features_gdf.drop('id', axis=1).to_file(self.vector_path, mode=self.c_write_mode, crs=self.crs)
            self.c_write_mode="a"
