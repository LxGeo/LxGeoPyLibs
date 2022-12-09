#

from LxGeoPyLibs.dataset.specific_datasets.raster_with_references import RasterWithRefsDataset
from LxGeoPyLibs.dataset.specific_datasets.rasterized_vector_with_reference import VectorWithRefDataset

DATASETS_REGISTERY = {
    
    "RasterWithRefsDataset": RasterWithRefsDataset,
    "VectorWithRefDataset": VectorWithRefDataset
    
}