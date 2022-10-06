from LxGeoPyLibs.dataset.patchified_dataset import PatchifiedDataset
from LxGeoPyLibs.dataset.raster_dataset import RasterDataset
from torch.utils.data import Dataset
import typing
from LxGeoPyLibs.vision.image_transformation import Trans_Identity
import pygeos
from collections import OrderedDict, defaultdict
import torch



class RasterWithRefsDataset(Dataset, PatchifiedDataset):
    """
    Dataset used to load raster and reference rasters.
    Example of use case: load two probability maps to predict optical flow.
    """

    def __init__(self, image_path:str, ref_images_dict:typing.OrderedDict[str, str], force_coord_transform=False,
     augmentation_transforms=None,preprocessing=None, ref_preprocessing: typing.DefaultDict[str, typing.Callable] = defaultdict(lambda x:x),bounds_geom=None, 
     patch_size=None, patch_overlap=None):
        """
        Args:
            -image_path: filepath of raster map file (tif)
            -ref_images_dict: ordered dict of reference rasters with keys as image_id and values as image paths
            -force_coord_transform: if True transforms geometries to raster crs if different
        """

        self.image_dataset = RasterDataset(image_path=image_path)
        self.refs_dataset_dict = {k: RasterDataset(image_path=ref_image_path) for k, ref_image_path in ref_images_dict.items()}

        if augmentation_transforms is None:
            self.augmentation_transforms=[Trans_Identity()]
        else:
            self.augmentation_transforms = augmentation_transforms
        self.preprocessing=preprocessing
        self.ref_preprocessing=ref_preprocessing

        crs_set = set([self.image_dataset.rio_dataset().crs] + [c_dataset.rio_dataset().crs for c_dataset in self.refs_dataset_dict.values()])
        crs_are_equal = len(crs_set)==1
        if not crs_are_equal:
            if force_coord_transform:
                print("Cannot force coords transformations! Not implemented yet")
                raise Exception("CRS mismatch.")
            else:
                print("Vector and raster inputs don't share the same crs!")
                print("Transform one of the inputs or change 'force_coord_transform' to True!")
                raise Exception("CRS mismatch.")
        
        common_area_geom = pygeos.intersection_all(
            [pygeos.box(*self.image_dataset.rio_dataset().bounds)] + [
                pygeos.box(*c_dataset.rio_dataset().bounds) for c_dataset in self.refs_dataset_dict.values()
                ])
        
        if pygeos.is_empty(common_area_geom):
            print("Vector and raster don't have common area!")
            raise Exception("Area mismatch.")
        
        if bounds_geom:
            common_area_geom = pygeos.intersection(common_area_geom, bounds_geom)
        
        self.bounds_geom=common_area_geom

        if not None in (patch_size, patch_overlap):
            self.setup_spatial(patch_size, patch_overlap, common_area_geom)
    
    #Refactor below (DRY)
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
        
        ref_imgs = {k: ref_dataset._load_padded_raster_window(window_geom, self.patch_size) for k, ref_dataset in self.refs_dataset_dict.items()}
        
        c_trans = self.augmentation_transforms[transform_idx]
        img, _, refs = c_trans(img, img, *ref_imgs.values())
        for k, t_ref_img in zip(ref_imgs.keys(), refs):
            c_preprocessor = self.ref_preprocessing[k]
            ref_imgs[k]=c_preprocessor(t_ref_img)
        
        if self.preprocessing:
            img = self.preprocessing(img)
        
        img = torch.from_numpy(img).float()
        for k,v in ref_imgs.items():
            ref_imgs[k]=torch.from_numpy(v).float()
        
        return img, *ref_imgs.values()

    def get_stacked_batch(self, input_to_stack):

        dezipped = list(zip(*input_to_stack))

        return [ torch.stack(d) for d in dezipped ]

    def gsd(self):
        return self.image_dataset.gsd()