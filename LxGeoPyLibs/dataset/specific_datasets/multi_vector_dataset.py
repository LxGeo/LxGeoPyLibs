from LxGeoPyLibs.dataset.patchified_dataset import PatchifiedDataset
from LxGeoPyLibs.dataset.vector_dataset import VectorDataset
from torch.utils.data import Dataset
import typing
from collections import OrderedDict, defaultdict
import tqdm
import pygeos


class MultiVectorDataset(Dataset, PatchifiedDataset):
    """
    Dataset used to load multiple vectors.
    Example of use case: load two vector maps in patchified way
    """

    def __init__(self, vectors_dict:typing.OrderedDict[str, str], force_coord_transform=False, augmentation_transforms=None,
     preprocessings:  typing.Union[ typing.DefaultDict[str, typing.Callable], typing.Callable] = defaultdict(lambda :lambda x:x),
     bounds_geom=None, spatial_patch_size=None, spatial_patch_overlap=None, in_fields=None, ex_fields=None):
        """
        Args:
            -vectors_dict: ordered dict of reference vectors with keys as vector_map_id and values as vector paths
            -force_coord_transform: if True transforms geometries to first vector map crs if different
            -augmentation_transforms: a set of callable to apply on a window view of geo dataframe 
            -preprocessings: a callable or dict of callables to apply on a window view of geo dataframe respectively to each dataset
            -bounds_geom: pygeos geometry (polygon) used as global bounds for the current dataset (must intersect with common area)
        """

        self.sub_datasets_dict = {k: VectorDataset(vector_path=c_vector_path) for k, c_vector_path in vectors_dict.items()}

        if augmentation_transforms is None:
            self.augmentation_transforms=[lambda x:x]
        else:
            self.augmentation_transforms = augmentation_transforms
        
        if isinstance(preprocessings, typing.Callable):
            self.preprocessings = defaultdict(lambda :preprocessings)
        else:
            self.preprocessings=preprocessings

        self.in_fields = in_fields
        self.ex_fields = ex_fields

        crs_set = set([c_dataset.crs() for c_dataset in self.sub_datasets_dict.values()])
        crs_are_equal = len(crs_set)==1
        if not crs_are_equal:
            if force_coord_transform:
                print("Cannot force coords transformations! Not implemented yet")
                raise Exception("CRS mismatch.")
            else:
                print("Vectors inputs don't share the same crs!")
                print("Transform one of the inputs or change 'force_coord_transform' to True!")
                raise Exception("CRS mismatch.")
        
        common_area_geom = pygeos.intersection_all(
            [pygeos.box(*c_dataset.fio_dataset().bounds) for c_dataset in self.sub_datasets_dict.values()]
            )
        
        if pygeos.is_empty(common_area_geom):
            print("Vectors don't have common area!")
            raise Exception("Area mismatch.")
        
        if bounds_geom:
            assert not pygeos.empty( pygeos.intersection(common_area_geom, bounds_geom) ), "bounds_geom doesn't intersect with common area"
            self.bounds_geom = bounds_geom
        else:
            self.bounds_geom=common_area_geom

        if not None in (spatial_patch_size, spatial_patch_overlap):
            self.setup_spatial(spatial_patch_size, spatial_patch_overlap, self.bounds_geom)
    
    def setup_spatial(self, patch_size_spatial, patch_overlap_spatial, bounds_geom):
        super().__init__(patch_size_spatial, patch_overlap_spatial, bounds_geom)
    
    def __getitem__(self, idx):
        
        assert self.is_setup, "Dataset is not set up!"
        window_idx = idx // (len(self.augmentation_transforms))
        transform_idx = idx % (len(self.augmentation_transforms))
        
        window_geom = super(Dataset, self).__getitem__(window_idx)
                
        loaded_vectors_dict = {
            k: ref_dataset._load_vector_features_window(window_geom, in_fields=self.in_fields, ex_fields=self.ex_fields)
             for k, ref_dataset in self.sub_datasets_dict.items()
            }
        
        c_trans = self.augmentation_transforms[transform_idx]

        transformed_vectors_dict = { k: c_trans(c_view) for k, c_view in loaded_vectors_dict.items()}
        
        for k, c_view in transformed_vectors_dict.items():
            c_preprocessor = self.preprocessings[k]
            transformed_vectors_dict[k]=c_preprocessor(c_view)                
                
        return transformed_vectors_dict

if __name__ == "__main__":
    
    vector_maps_dict = {
        "in_vector1" :"../DATA_SANDBOX/lxFlowAlign/data/raw/paris/DL/ortho1/rooftop/buildings.shp",
        "in_vector2" : "../DATA_SANDBOX/lxFlowAlign/data/raw/paris/DL/ortho2/rooftop/buildings.shp"
    }
    c_v = MultiVectorDataset(vector_maps_dict, spatial_patch_size=(512,512),spatial_patch_overlap=10)
    for i in tqdm.tqdm(range(len(c_v))):
        geom = c_v[i]
    print(geom)
    pass