from inspect import Attribute
from LxGeoPyLibs.dataset.raster_dataset import RasterDataset, rasters_map
from LxGeoPyLibs.geometry.utils_pygeos import get_pygeos_geom_creator, get_pygeos_transformer
import fiona
import pyproj
import pygeos

class VectorWithRefDataset(RasterDataset):
    """
    Dataset used to load vector and raster data.
    Example of use case: load non aligned vector map with respective reference map as a raster of probability.
    """

    def __init__(self, image_path:str, vector_path:str, force_coord_transform=False, **kwargs):
        """
        Args:
            -image_path: filepath of raster map file (tif)
            -vector_path: filepath of vector map file (shp, TAB)
            -force_coord_transform: if True transforms geometries to raster crs if different
        """
        RasterDataset.__init__(self, image_path=image_path, **kwargs)
        self.vector_dataset = fiona.open(vector_path)
        
        # check crs correspondance
        crs_are_equal = self.vector_dataset.crs == rasters_map[self.image_path].crs
        if not crs_are_equal:
            if not force_coord_transform:
                print("Vector and raster inputs don't share the same crs!")
                print("Transform one of the inputs or change 'force_coord_transform' to True!")
                raise Exception("CRS mismatch.")
            
        projection_transformer = pyproj.Transformer.from_proj(
            pyproj.Proj(init=self.vector_dataset.crs["init"]),
            pyproj.Proj(init=rasters_map[self.image_path].crs)
        )
        pygeos_transformer = get_pygeos_transformer(projection_transformer)

        vector_bounds_geom = pygeos_transformer(pygeos.box(*self.vector_dataset.bounds))
        raster_bounds_geom = pygeos.box(*rasters_map[self.image_path].bounds)
        common_area_geom = pygeos.intersection(vector_bounds_geom,raster_bounds_geom)
        if pygeos.is_empty(common_area_geom):
            print("Vector and raster don't have common area!")
            raise Exception("Area mismatch.")
        
        patch_size=kwargs.get("patch_size")
        patch_overlap=kwargs.get("patch_overlap")
        if not None in (patch_size, patch_overlap):
            self.setup_spatial(patch_size, patch_overlap, common_area_geom)
    
    
    def vector_geometry_type(self):
        return self.vector_dataset.meta["schema"]["geometry"]

    def _load_vector_geometries_window(self, window_geom, crop=True):
        """
        Function to load geometries from vector within a window
        """

        features_coords = []
        for c_feature in self.vector_dataset.filter(bbox=pygeos.bounds(window_geom).tolist()):
            features_coords.append( c_feature["geometry"]["coordinates"] )
        
        pygeos_geom_creator = get_pygeos_geom_creator(self.vector_geometry_type())

        geometries = pygeos_geom_creator(features_coords)

        if crop: geometries=pygeos.intersection(geometries, window_geom)

        return geometries

if __name__ == "__main__":

    in_raster = "../DATA_SANDBOX/lxFlowAlign/data/train_data/paris_ortho1_rooftop/flow.tif"
    in_vector = "../DATA_SANDBOX/lxFlowAlign/data/raw/paris/DL/ortho1/rooftop/buildings.shp"
    c_r = VectorWithRefDataset(in_raster,in_vector, patch_size=(256,256), patch_overlap=0 )

    a=c_r._load_vector_geometries_window(c_r.patch_grid[20])
    print(a)
    pass



