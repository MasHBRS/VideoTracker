import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

class Composition:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data=None):
        for t in self.transform:
                if (torch.is_tensor(data)):
                    data= t(data)
                elif len(data)>1:
                    data= t(*data)
                else :
                    data= t(data)
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

    def __call__(self, bbox=None, mask=None, rgb=None):
        
        if random.random() < self.p:
            rgb = F.vflip(rgb)
            if (bbox is None and mask is None):
                return rgb
            mask = F.vflip(mask)

            Height = rgb.shape[2]  # 128
            
            # Step 1: Adjust bounding box coordinates
            flipped_bboxes = bbox.clone()  # Create a copy to avoid modifying the original
            flipped_bboxes[:,:, 1] = Height - bbox[:,:, 3]  # New y_min = H - y_max
            flipped_bboxes[:,:, 3] = Height - bbox[:,:, 1]  # New y_max = H - y_min
            # x_min (index 0) and x_max (index 2) remain unchanged
            
            # Optional: Ensure coordinates are within bounds [0, H]
            flipped_bboxes = torch.clamp(flipped_bboxes, 0, Height)

            #com[:,:, 1] = Height - com[:,:, 1]
            # Optional: Ensure coms are within bounds [0, H]
            #com = torch.clamp(com, 0, Height)
            return flipped_bboxes,mask,rgb
        if (bbox is None and mask is None):
            return rgb
        return bbox,mask,rgb

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

    def __call__(self, bbox, mask, rgb):
        
        if random.random() < self.p:
            rgb = F.hflip(rgb)
            if (bbox is None and mask is None):
                return rgb
            mask = F.hflip(mask)

            Width = rgb.shape[3]  # 128
            
            # Step 1: Adjust bounding box coordinates
            flipped_bboxes = bbox.clone()  # Create a copy to avoid modifying the original
            flipped_bboxes[:,:, 0] = Width - bbox[:,:, 2]  # New x_min = H - x_max
            flipped_bboxes[:,:, 2] = Width - bbox[:,:, 0]  # New x_max = H - x_min
            # y_min (index 0) and y_max (index 2) remain unchanged
            
            # Optional: Ensure coordinates are within bounds [0, H]
            flipped_bboxes = torch.clamp(flipped_bboxes, 0, Width)

            #com[:,:, 0] = Width - com[:,:, 0]
            # Optional: Ensure coms are within bounds [0, H]
            #com = torch.clamp(com, 0, Width)
            return flipped_bboxes,mask,rgb
        if (bbox is None and mask is None):
            return rgb
        return bbox,mask,rgb

class CustomResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, rgb,bboxs=None,masks=None):
        
        scale_height = self.size[0] / rgb.shape[2]
        scale_width = self.size[1] / rgb.shape[3]
        
        resizer=T.Resize(self.size)
        new_rgb = resizer(rgb)
        if (bboxs is None and masks is None):
            return rgb
        new_masks = resizer(masks)
        
        resized_bboxes = bboxs.clone()
        resized_bboxes[..., [0, 2]] = resized_bboxes[..., [0, 2]] * scale_width  # xmin, xmax
        resized_bboxes[..., [1, 3]] = resized_bboxes[..., [1, 3]] * scale_height  # ymin, ymax

        #resized_coms = coms.clone()
        #resized_coms[..., 1] = resized_coms[..., 1] * scale_width  # xmin, xmax
        #resized_coms[..., 0] = resized_coms[..., 0] * scale_height  # ymin, ymax

        # TODO: coms, bboxes are left
        return resized_bboxes,new_masks,new_rgb

class CustomColorJitter:
    def __init__(self, brightness, hue, contrast, saturation):
        self.brightness_factor = brightness
        self.contrast_factor = contrast    
        self.saturation_factor = saturation  
        self.hue_factor = hue        
        
    def __call__(self, bboxs,masks,rgb ):
        brightness_factor = random.uniform(*self.brightness_factor)
        contrast_factor   = random.uniform(*self.contrast_factor)
        saturation_factor = random.uniform(*self.saturation_factor)
        hue_factor        = random.uniform(*self.hue_factor)
        
        #rgbs shape: [B,C,H,W]
        augmented_rgbs = torch.stack([
            F.adjust_hue(
                F.adjust_saturation(
                    F.adjust_contrast(
                        F.adjust_brightness(img, brightness_factor), 
                        contrast_factor
                    ), 
                saturation_factor
            ), hue_factor
        )
        for img in rgb])
        return bboxs,masks,augmented_rgbs

class RGBNormalizer:
    def __init__(self):
        pass
    def __call__(self, rgb=None,bboxs=None,masks=None ):
        if  bboxs==None and masks==None:
            return rgb/255
        else: 
            return bboxs,masks, rgb/255
class RGBImageBasedNormalizer:
    def __init__(self):
        pass
    def __call__(self, rgb=None):
        return rgb/255