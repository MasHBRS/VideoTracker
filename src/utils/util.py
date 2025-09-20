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
        msks=torch.zeros(size=(max_objects_in_scene,*masks.shape),device=imgs.device) # shape=[max_objects_in_scene, H, W]
        for uc in masks.unique():
            output[uc.item()]=imgs*(masks==uc)
            msks[uc.item()]=(masks==uc)
        return output,msks 

    def extract_bboxed_images(self,imgs,bboxes,max_objects_in_scene)-> torch.tensor:
        assert imgs.dim()==3 # imgs.shape=[C,H,W]

        output=torch.zeros(size=(max_objects_in_scene,*imgs.shape),device=imgs.device) # shape=[max_objects_in_scene, C, H, W]
        output[0]=imgs
        for idx,box in enumerate(bboxes):
            if torch.all(box == -1):
                continue
            output[idx+1,:,int(box[1].item()):int(box[3].item()),int(box[0].item()):int(box[2].item())]=imgs[:,int(box[1].item()):int(box[3].item()),int(box[0].item()):int(box[2].item())]
            output[0,:,int(box[1].item()):int(box[3].item()),int(box[0].item()):int(box[2].item())]=0
        return output