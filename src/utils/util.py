"""
TODO:
1. add background extract as well to the output 
"""
import torch


class ImageExtractor():
    def __init__(self)-> None:
        pass
    def extract_masked_images(self,imgs,masks,max_objects_in_scene)-> torch.Tensor:
        assert imgs.dim()==3 # imgs.shape=[C,H,W]
        
        output=torch.zeros(size=(max_objects_in_scene,*imgs.shape),device=imgs.device) # shape=[max_objects_in_scene, C, H, W]
        for uc in masks.unique():
            output[uc.item()]=imgs*(masks==uc)
        return output 
    def extract_bboxed_images(self,imgs,bboxes,max_objects_in_scene)-> torch.tensor:
        assert imgs.dim()==3 # imgs.shape=[C,H,W]

        output=torch.zeros(size=(max_objects_in_scene,*imgs.shape),device=imgs.device) # shape=[max_objects_in_scene, C, H, W]
        for idx,box in enumerate(bboxes):
            if torch.all(box == -1):
                continue
            output[idx,:,int(box[1].item()):int(box[3].item()),int(box[0].item()):int(box[2].item())]=imgs[:,int(box[1].item()):int(box[3].item()),int(box[0].item()):int(box[2].item())]
        return output