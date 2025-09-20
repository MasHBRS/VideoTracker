"""
TODO:
1. add background extract as well to the output 
"""
import torch


class ImageExtractor():
    def __init__(self) -> None:
        pass
    def extract_masked_images(self,imgs,masks,max_objects_in_scene)-> torch.Tensor:
        assert imgs.dim()==3 # imgs.shape=[C,H,W]

        output=torch.zeros(size=(max_objects_in_scene,*imgs.shape),device=imgs.device) # shape=[mask_number, C, H, W]
        for uc in masks.unique():
            output[uc.item()]=imgs*(masks==uc)
        return output 
    def extract_bboxed_images(self,imgs,boxed_imgs)-> torch.tensor:
        pass