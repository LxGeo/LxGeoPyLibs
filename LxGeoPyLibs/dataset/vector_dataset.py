
from LxGeoPyLibs.dataset.patchified_dataset import PatchifiedDataset
import multiprocessing
import fiona
import pygeos
import shapely.geometry
from collections.abc import Iterable
from collections import defaultdict
from LxGeoPyLibs.geometry.utils_pygeos import get_pygeos_geom_creator
import geopandas as gpd

class VectorRegister(dict):

    def __init__(self):
        super(VectorRegister, self).__init__()
    
    def __del__(self):
        for k,v in self.items():
            print("Closing vector at {}".format(k))
            v.close()

vectors_map=VectorRegister()
lock = multiprocessing.Lock()

class VectorDataset(PatchifiedDataset):

    def __init__(self, vector_path:str, augmentation_transforms=None, preprocessing=None, bounds_geom=None, spatial_patch_size=None, spatial_patch_overlap=None):

        self.vector_path = vector_path

        vectors_map.update({
            self.vector_path: fiona.open(self.vector_path)
            })
        
        vector_total_bounds_geom = pygeos.box(*self.fio_dataset().bounds)
        if bounds_geom:
            assert pygeos.intersects(vector_total_bounds_geom, bounds_geom), "Boundary geometry is out of vector extents!"
            self.bounds_geom = bounds_geom
        else:
            self.bounds_geom = vector_total_bounds_geom
        
        self.preprocessing=preprocessing
        self.is_setup=False

        if augmentation_transforms is None:
            self.augmentation_transforms=[lambda x:x]
        else:
            self.augmentation_transforms = augmentation_transforms

        if not None in (spatial_patch_size, spatial_patch_overlap):
            self.setup_spatial(spatial_patch_size, spatial_patch_overlap, self.bounds_geom)
        
    def fio_dataset(self):
        return vectors_map[self.vector_path]
    
    def vector_geometry_type(self):
        return self.fio_dataset().meta["schema"]["geometry"]
        
    def setup_spatial(self, patch_size_spatial, patch_overlap_spatial, bounds_geom):
        """
        Setup patch loading settings using spatial coordinates.
        Args:
            patch_size: a tuple of positive integers in coords metric.
            patch_overlap: a positive integer in coords metric.
            bounds_geom: pygeos polygon
        """
        super().__init__(patch_size_spatial, patch_overlap_spatial, bounds_geom)
    
    def _load_vector_geometries_window(self, window_geom, crop=False):
        """
        Function to load geometries from vector within a window
        Args:
            window_geom: pygeos geometry window
            crop: boolean to crop requested geometries within window_geom
        """
        lock.acquire()
        
        features_coords = []
        for c_feature in self.fio_dataset().filter(bbox=pygeos.bounds(window_geom).tolist()):
            if c_feature["geometry"]["type"].lower().startswith("multi"):
                features_coords.extend(c_feature["geometry"]["coordinates"])
            else:
                features_coords.append( c_feature["geometry"]["coordinates"] )
        
        lock.release()
        pygeos_geom_creator = get_pygeos_geom_creator(self.vector_geometry_type())

        geometries = pygeos_geom_creator(features_coords)

        if crop: geometries=pygeos.intersection(geometries, window_geom)

        return geometries
    
    def _load_vector_features_window(self, window_geom, ignore_multipart=False, in_fields=None, ex_fields=None):
        """
        Method to load features (geometry + fields) from vector map within a window
        Args:
            window_geom: pygeos geometry window
            in_fields: an iterable of included field names (None : All fields)
            ex_fields: an iterable of excluded field names.
        """
        if type(window_geom) ==  pygeos.Geometry:
            window_bounds = pygeos.bounds(window_geom).tolist()
        elif type(window_geom )== shapely.geometry.polygon.Polygon:
            window_bounds = window_geom.bounds
        elif isinstance(window_geom, Iterable):
            assert len(window_geom)==4, f"Window geometry of type {type(window_geom)} has more than 4 values!"
            window_bounds = list(window_geom)

        if in_fields is None:
            fields_names_set = set(self.fio_dataset().schema['properties'].keys())
        else:
            fields_names_set = set(in_fields)

        if ex_fields is None:
            ex_fields=[]

        fields_names_set=fields_names_set.difference(set(ex_fields))

        features_coords = []
        columns = defaultdict(list)
        for c_feature in self.fio_dataset().filter(bbox=window_bounds):
            multipart_count=1
            if c_feature["geometry"]["type"].lower().startswith("multi"):
                if ignore_multipart: continue
                features_coords.extend(c_feature["geometry"]["coordinates"])
                multipart_count=len(c_feature["geometry"]["coordinates"])
            else:
                features_coords.append( c_feature["geometry"]["coordinates"] )

            columns["id"].extend([c_feature["id"]]*multipart_count)

            for k,v in c_feature["properties"].items():
                if k in fields_names_set:
                    columns[k].extend([v]*multipart_count)

        pygeos_geom_creator = get_pygeos_geom_creator(self.vector_geometry_type())

        columns["geometry"] = pygeos_geom_creator(features_coords)

        gdf = gpd.GeoDataFrame(columns)
        return gdf
    
    def __getitem__(self, idx):        
        assert self.is_setup, "Dataset is not set up!"
        window_idx = idx // (len(self.augmentation_transforms))
        transform_idx = idx % (len(self.augmentation_transforms))        
        window_geom = super().__getitem__(window_idx)
        requested_geoms = self._load_vector_geometries_window(window_geom)
        c_trans = self.augmentation_transforms[transform_idx]
        transformed_geoms = c_trans(requested_geoms)
        return transformed_geoms


if __name__ == "__main__":
    in_vector = "../DATA_SANDBOX/lxFlowAlign/data/raw/paris/DL/ortho1/rooftop/buildings.shp"
    c_v = VectorDataset(in_vector, spatial_patch_size=(512,512),spatial_patch_overlap=100)
    geom = c_v[10]
    print(geom)
    pass

