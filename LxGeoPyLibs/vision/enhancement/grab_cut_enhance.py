

from collections import OrderedDict
import numpy as np
import cv2
from LxGeoPyLibs.dataset.specific_datasets.raster_with_references import RasterWithRefsDataset


class test_model():

    device = "cpu"

    def __init__(self) -> None:
        pass

    def __call__(self, x):
        if hasattr("forward"):
            return self.forward(x)
        return x
    
    def batch_size(self):
        return 1
    
    def min_patch_size(self):
        return 128
    
    def predict_step(self, batch, batch_idx: int = None, dataloader_idx: int = 0):
        return self.forward(batch)

def proba2mask(proba_array):
    """
    Works only for 3bands undersegmented proba of buildings!!
    """

    labels_array = np.argmax(proba_array, -1)

    # init grab mask as all probable foreground pixels 
    grab_mask = cv2.GC_PR_FGD*np.ones(proba_array.shape[:-1])

    # probable background
    pbg = np.logical_or(proba_array[:,:,0]>=128, proba_array[:,:,1]>=128)
    grab_mask[pbg]=cv2.GC_PR_BGD

    # background confiremed values || update grab_mask
    bg_mask = np.logical_or(proba_array[:,:,0]>=190, proba_array[:,:,1]>=190)
    grab_mask[bg_mask]=cv2.GC_BGD
    # foreground confirmed values || update grab_mask
    fg_mask = labels_array>1
    grab_mask[fg_mask]=cv2.GC_PR_FGD
    
    return grab_mask




class grabCutModel(test_model):

    def __init__(self, mask_preprocessor=proba2mask):
        self.mask_preprocessor = mask_preprocessor

    def forward(self, image_pair):

        outputs=[]
        image_batch, proba_batch = image_pair
        for c_idx in range(image_batch.shape[0]):
            outputs.append(self.forward_one( (image_batch[c_idx], proba_batch[c_idx]) ))
        return np.stack(outputs, axis=0)
        
    
    def forward_one(self, image_pair):

        image, proba = image_pair
        # torch 2 numpy
        image = image.permute(1,2,0).numpy().astype(np.uint8)
        proba = proba.permute(1,2,0).numpy()

        mask = self.mask_preprocessor(proba).astype(np.uint8)

        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        mask, bgdModel, fgdModel = cv2.grabCut(image,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

        return mask

if __name__ == "__main__":

    gc = grabCutModel()

    in_raster_path= "../../../DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/LA/oneAtlas_catalogs_extracted/PHR1A_acq20220728_del1070b795/ortho.tif"
    in_raster_path2="../../../DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/LA/oneAtals_catalogs_prediction/PHR1A_acq20220728_del1070b795/build_probas.tif"
    in_dataset = RasterWithRefsDataset(in_raster_path, OrderedDict(im2=in_raster_path2))

    out_path = "grab_mask.tif"
    in_dataset.predict_to_file(out_path, gc, tile_size=(3000,3000))
