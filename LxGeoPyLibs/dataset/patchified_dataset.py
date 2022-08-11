
from torch.utils.data import Dataset
from LxGeoPyLibs.geometry.grid import make_grid
import pygeos

class PatchifiedDataset(object):
    """
    
    """

    def __init__(self, spatial_patch_size, spatial_patch_overlap, bounds_geom):
        super().__init__()

        """
        Setup patch loading settings using spatial coordinates.
        Args:
            patch_size: a couple of positive integers in spatial metric.
            patch_overlap: a positive integer in spatial metric.
            bounds_geom: pygeos polygon
        """

        self.spatial_patch_size = spatial_patch_size
        self.spatial_patch_overlap = spatial_patch_overlap
        self.bounds_geom = bounds_geom
        # buffer bounds_geom to include out of bound area
        buff_bounds_geom = pygeos.buffer(bounds_geom, self.spatial_patch_overlap, cap_style="square", join_style="mitre")

        grid_step = self.spatial_patch_size[0]-self.spatial_patch_overlap*2, self.spatial_patch_size[1]-self.spatial_patch_overlap*2
        assert grid_step[0]>0 and grid_step[1]>0 , "Spatial patch overlap is high! Reduce patch overlap."
        self.patch_grid = make_grid(buff_bounds_geom, grid_step[0], grid_step[1], self.spatial_patch_size[0], self.spatial_patch_size[1],
                                    filter_predicate = lambda x: pygeos.intersects(x, bounds_geom) )
        
        self.is_setup=True
    
    def __len__(self):
        return len(self.patch_grid)
    
    def __getitem__(self, index):
        return self.patch_grid[index]
