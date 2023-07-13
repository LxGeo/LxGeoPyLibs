#

from LxGeoPyLibs.dataset.specific_datasets.raster_with_references import RasterWithRefsDataset
from LxGeoPyLibs.dataset.specific_datasets.rasterized_vector_with_reference import VectorWithRefDataset
from LxGeoPyLibs.dataset.raster_dataset import RasterDataset

DATASETS_REGISTERY = {
    
    "RasterWithRefsDataset": RasterWithRefsDataset,
    "VectorWithRefDataset": VectorWithRefDataset,
    "RasterDataset": RasterDataset
    
}