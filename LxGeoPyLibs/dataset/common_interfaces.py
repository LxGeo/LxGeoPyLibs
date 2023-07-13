from LxGeoPyLibs.dataset.patchified_dataset import PatchifiedDataset

class SpatialyProjectedDataset(object):
    def __init__(self, crs):
        self._crs=crs
    def crs(self):
        return self._crs

class BoundedDataset(object):

    def __init__(self, bounds_geom):
        self.bounds_geom = bounds_geom

class PixelizedDataset(PatchifiedDataset):

    def __init__(self, pixel_x_size, pixel_y_size):
        self.pixel_x_size=pixel_x_size
        self.pixel_y_size=pixel_y_size

    ### should be fixed to meters not pixels
    def setup_patch_per_pixel(self, pixel_patch_size, pixel_patch_overlap, bounds_geom):
        """
        Setup patch loading settings using spatial coordinates.
        Args:
            patch_size: a tuple of positive integers in pixels.
            patch_overlap: a positive integer in pixels.
            bounds_geom: pygeos polygon
        """
        self.pixel_patch_size= pixel_patch_size
        self.pixel_patch_overlap= pixel_patch_overlap

        patch_size_spatial = (self.pixel_patch_size[0]*self.pixel_x_size, self.pixel_patch_size[1]*self.pixel_y_size)
        patch_overlap_spatial = self.pixel_patch_overlap*self.pixel_x_size

        PatchifiedDataset.__init__(self, patch_size_spatial, patch_overlap_spatial, bounds_geom)


