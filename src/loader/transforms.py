import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

class Composition:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data):
        
        for t in self.transform:
                data= t(*data)
        if(len(self.transform)==0):
            return []
        return data
        
class RandomVerticalFlip:
    """
    Randomly flips the image frame vertically with a probability p.
    Args:
        p (float): Probability of the image being flipped.
        
        com:  torch.Size([11, 2])
        bbox: torch.Size([11, 4])
        mask: torch.Size([128, 128])
        rgb: torch.Size([3, 128, 128])
        flow: torch.Size([3, 128, 128])
        torch.Size([11, 4]) torch.Size([128, 128]) torch.Size([3, 128, 128]) torch.Size([3, 128, 128])
    """
    
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, com, bbox, mask, rgb, flow):
        
        if random.random() < self.p:
            rgb = F.vflip(rgb)
            flow = F.vflip(flow)
            mask = F.vflip(mask)

            Height = rgb.shape[2]  # 128
            
            # Step 1: Adjust bounding box coordinates
            flipped_bboxes = bbox.clone()  # Create a copy to avoid modifying the original
            flipped_bboxes[:,:, 1] = Height - bbox[:,:, 3]  # New y_min = H - y_max
            flipped_bboxes[:,:, 3] = Height - bbox[:,:, 1]  # New y_max = H - y_min
            # x_min (index 0) and x_max (index 2) remain unchanged
            
            # Optional: Ensure coordinates are within bounds [0, H]
            flipped_bboxes = torch.clamp(flipped_bboxes, 0, Height)

            com[:,:, 1] = Height - com[:,:, 1]
            # Optional: Ensure coms are within bounds [0, H]
            com = torch.clamp(com, 0, Height)
            return com,flipped_bboxes,mask,rgb,flow

        return com,bbox,mask,rgb,flow

class RandomHorizontalFlip:
    """
    Randomly flips the image frame Horizontally with a probability p.
    Args:
        p (float): Probability of the image being flipped.
        
        com:  torch.Size([11, 2])
        bbox: torch.Size([11, 4])
        mask: torch.Size([128, 128])
        rgb: torch.Size([3, 128, 128])
        flow: torch.Size([3, 128, 128])
        torch.Size([11, 4]) torch.Size([128, 128]) torch.Size([3, 128, 128]) torch.Size([3, 128, 128])
    """
    
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, com, bbox, mask, rgb, flow):
        
        if random.random() < self.p:
            rgb = F.hflip(rgb)
            flow = F.hflip(flow)
            mask = F.hflip(mask)

            Width = rgb.shape[3]  # 128
            
            # Step 1: Adjust bounding box coordinates
            flipped_bboxes = bbox.clone()  # Create a copy to avoid modifying the original
            flipped_bboxes[:,:, 0] = Width - bbox[:,:, 2]  # New x_min = H - x_max
            flipped_bboxes[:,:, 2] = Width - bbox[:,:, 0]  # New x_max = H - x_min
            # y_min (index 0) and y_max (index 2) remain unchanged
            
            # Optional: Ensure coordinates are within bounds [0, H]
            flipped_bboxes = torch.clamp(flipped_bboxes, 0, Width)

            com[:,:, 0] = Width - com[:,:, 0]
            # Optional: Ensure coms are within bounds [0, H]
            com = torch.clamp(com, 0, Width)
            return com,flipped_bboxes,mask,rgb,flow
        return com,bbox,mask,rgb,flow

class CustomResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, coms,bboxs,masks,rgbs,flows ):
        
        scale_height = self.size[0] / rgbs.shape[2]
        scale_width = self.size[1] / rgbs.shape[3]
        
        resizer=T.Resize(self.size)
        new_rgb = resizer(rgbs)
        new_flows = resizer(flows)
        new_masks = resizer(masks)
        
        resized_bboxes = bboxs.clone()
        resized_bboxes[..., [0, 2]] = resized_bboxes[..., [0, 2]] * scale_width  # xmin, xmax
        resized_bboxes[..., [1, 3]] = resized_bboxes[..., [1, 3]] * scale_height  # ymin, ymax

        resized_coms = coms.clone()
        resized_coms[..., 1] = resized_coms[..., 1] * scale_width  # xmin, xmax
        resized_coms[..., 0] = resized_coms[..., 0] * scale_height  # ymin, ymax


        # TODO: coms, bboxes are left
        return resized_coms,resized_bboxes,new_masks,new_rgb,new_flows