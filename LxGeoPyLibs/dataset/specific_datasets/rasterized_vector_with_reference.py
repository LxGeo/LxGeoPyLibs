from LxGeoPyLibs.dataset.raster_dataset import RasterDataset
from LxGeoPyLibs.dataset.vector_dataset import VectorDataset
from LxGeoPyLibs.dataset.patchified_dataset import PatchifiedDataset
from torch.utils.data import Dataset
from LxGeoPyLibs.geometry.utils_pygeos import get_pygeos_geom_creator, get_pygeos_transformer
import fiona
import pyproj
import pygeos
import torch
from LxGeoPyLibs.vision.image_transformation import Trans_Identity
from LxGeoPyLibs.geometry.rasterizers.polygons_rasterizer import polygons_to_multiclass

class VectorWithRefDataset(Dataset, PatchifiedDataset):
    """
    Dataset used to load vector and raster data.
    Example of use case: load non aligned vector map with respective reference map as a raster of probability.
    """

    def __init__(self, image_path:str, vector_path:str, rasterization_method, force_coord_transform=False,
     augmentation_transforms=None,preprocessing=None, bounds_geom=None, 
     patch_size=None, patch_overlap=None):
        """
        Args:
            -image_path: filepath of raster map file (tif)
            -vector_path: filepath of vector map file (shp, TAB)
            -rasterization_method: Callable that transforms vectors to rasters
            -force_coord_transform: if True transforms geometries to raster crs if different
        """
        self.image_dataset = RasterDataset(image_path=image_path)
        self.vector_dataset = VectorDataset(vector_path=vector_path)

        self.rasterization_method = rasterization_method
        if augmentation_transforms is None:
            self.augmentation_transforms=[Trans_Identity()]
        else:
            self.augmentation_transforms = augmentation_transforms
        self.preprocessing=preprocessing
        
        # check crs correspondance
        crs_are_equal = self.image_dataset.rio_dataset().crs == self.vector_dataset.fio_dataset().crs
        if not crs_are_equal:
            if not force_coord_transform:
                print("Vector and raster inputs don't share the same crs!")
                print("Transform one of the inputs or change 'force_coord_transform' to True!")
                raise Exception("CRS mismatch.")
            
        projection_transformer = pyproj.Transformer.from_proj(
            pyproj.Proj(init=self.vector_dataset.fio_dataset().crs["init"]),
            pyproj.Proj(init=self.image_dataset.rio_dataset().crs)
        )
        pygeos_transformer = get_pygeos_transformer(projection_transformer)

        vector_bounds_geom = pygeos_transformer(pygeos.box(*self.vector_dataset.fio_dataset().bounds))
        raster_bounds_geom = pygeos.box(*self.image_dataset.rio_dataset().bounds)
        common_area_geom = pygeos.intersection(vector_bounds_geom,raster_bounds_geom)
        if pygeos.is_empty(common_area_geom):
            print("Vector and raster don't have common area!")
            raise Exception("Area mismatch.")
        
        if bounds_geom:
            common_area_geom = pygeos.intersection(common_area_geom, bounds_geom)
        
        self.bounds_geom=common_area_geom
        
        if not None in (patch_size, patch_overlap):
            self.setup_spatial(patch_size, patch_overlap, common_area_geom)
        
    def setup_spatial(self, patch_size, patch_overlap, bounds_geom=None):
        """
        Setup patch loading settings using spatial coordinates.
        Args:
            patch_size: a tuple of positive integers in pixels.
            patch_overlap: a positive integer in pixels.
            bounds_geom: pygeos polygon
        """
        self.patch_size= patch_size
        self.patch_overlap= patch_overlap
        
        # If no bounds provided, use image bounds
        if not bounds_geom:
            bounds_geom = self.bounds_geom

        pixel_x_size = self.image_dataset.rio_dataset().transform[0]
        pixel_y_size = -self.image_dataset.rio_dataset().transform[4]

        patch_size_spatial = (self.patch_size[0]*pixel_x_size, self.patch_size[1]*pixel_y_size)
        patch_overlap_spatial = self.patch_overlap*pixel_x_size

        super(Dataset, self).__init__(patch_size_spatial, patch_overlap_spatial, bounds_geom)
    
    def __getitem__(self, idx):
        
        assert self.is_setup, "Dataset is not set up!"
        window_idx = idx // (len(self.augmentation_transforms))
        transform_idx = idx % (len(self.augmentation_transforms))
        
        window_geom = super(Dataset, self).__getitem__(window_idx)
        
        img = self.image_dataset._load_padded_raster_window(window_geom, self.patch_size)
        vec = self.vector_dataset._load_vector_geometries_window(window_geom)
        # rasterio accepts only features that implements __geo_interface__
        vec = list(map(lambda x: pygeos.to_shapely(x), vec))
        rasterized_vec = self.rasterization_method(vec, window_geom, gsd=self.image_dataset.gsd(), crs=self.image_dataset.rio_dataset().crs)
        
        c_trans = self.augmentation_transforms[transform_idx]
        img, rasterized_vec, _ = c_trans(img, rasterized_vec)
        
        if self.preprocessing:
            img = self.preprocessing(img)
        
        img = torch.from_numpy(img).float()
        rasterized_vec = torch.from_numpy(rasterized_vec).float()
        return img, rasterized_vec
    
    def get_stacked_batch(self, input_to_stack):

        dezipped = list(zip(*input_to_stack))

        return [ torch.stack(d) for d in dezipped ]

    def gsd(self):
        return self.image_dataset.gsd()
    

if __name__ == "__main__":

    in_raster = "../DATA_SANDBOX/lxFlowAlign/data/train_data/paris_ortho1_rooftop/flow.tif"
    in_vector = "../DATA_SANDBOX/lxFlowAlign/data/raw/paris/DL/ortho1/rooftop/buildings.shp"
    rasterization_method = polygons_to_multiclass
    c_r = VectorWithRefDataset(in_raster,in_vector, rasterization_method=rasterization_method, patch_size=(256,256), patch_overlap=0 )

    a=c_r[20]
    print(a)
    pass



