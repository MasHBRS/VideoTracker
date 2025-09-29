import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import io
import glob

class VideoDataset(Dataset):
    
    def __init__(self, data_path, max_objects_in_scene,halve_dataset,is_test_dataset, selected_number_of_frames_per_video=24,original_number_of_frames_per_video=24, split='train',transforms=None,device='cpu'):
        data_directory=os.path.join(data_path, split)
        if not os.path.exists(data_directory):
            if not os.path.exists(os.path.abspath(data_directory)):
                raise Exception("Dataset was not found!")
        self.selected_number_of_frames_per_video=selected_number_of_frames_per_video
        self.original_number_of_frames_per_video=original_number_of_frames_per_video
        self.device=device
        self.max_objects_in_scene=max_objects_in_scene
        self.transforms=transforms
        self.coord_addresses=sorted(glob.glob(os.path.join(data_directory,'coords*.pt')))
        self.mask_addresses=sorted(glob.glob(os.path.join(data_directory,'mask*.pt')))
        self.rgb_addresses=sorted(glob.glob(os.path.join(data_directory,'rgb*.png')))
        if(selected_number_of_frames_per_video!=original_number_of_frames_per_video):
            self.step=original_number_of_frames_per_video//selected_number_of_frames_per_video
            self.rgb_addresses=self.skipframes(rgbs=self.rgb_addresses)
        #self.flow_addresses=sorted(glob.glob(os.path.join(data_directory,'flow*.png')))
        self.halve_dataset=halve_dataset
        self.is_test_dataset=is_test_dataset

        self.number_of_videos=len(self.rgb_addresses)/selected_number_of_frames_per_video
        #print(f"Data Loaded Successfully: {len(self.coord_addresses)=}, {len(self.mask_addresses)=}, {len(self.rgb_addresses)=}, {len(self.flow_addresses)=}")

    def __len__(self):
        return int(self.number_of_videos) if self.halve_dataset==False else  int(self.number_of_videos)//2

    def __getitem__(self, filmIndex): # I should apply self.transforms here: to all video frames of each video. For instance: transforms.RandomHorizontalFlip(), transforms.RandomRotation(degrees=25), transforms.ColorJitter(brightness=.5, hue=.2, contrast=0.2, saturation=0.2), 
        if self.halve_dataset and self.is_test_dataset:
                filmIndex+=self.__len__()
        bboxs,masks,rgbs=self.read_data(filmIndex)
        if self.transforms!=None:
            bboxs,masks,rgbs = self.transforms((bboxs,masks,rgbs))
        return bboxs,masks,rgbs

    def read_data(self,filmIndex):
        coords_to_read=self.coord_addresses[filmIndex]
        masks_to_read=self.mask_addresses[filmIndex]
        rgbs_to_read=self.rgb_addresses[filmIndex*self.selected_number_of_frames_per_video:(filmIndex+1)*self.selected_number_of_frames_per_video]
        #flows_to_read=self.flow_addresses[filmIndex*self.selected_number_of_frames_per_video:(filmIndex+1)*self.selected_number_of_frames_per_video]

        coords_read=torch.load(coords_to_read, map_location="cpu") 
        masks_read=torch.load(masks_to_read, map_location="cpu") 
        rgbs_read=[io.read_image(path=file).to(torch.float32) for file in rgbs_to_read]
        #flows_read=[io.read_image(path=file).to(torch.float32) for file in flows_to_read]

        #coms_read_torch=coords_read['com']
        bbox_read_torch=coords_read['bbox'] if self.selected_number_of_frames_per_video==self.original_number_of_frames_per_video else coords_read['bbox'][0::self.step,:,:]
        masks_read_torch=masks_read['masks'] if self.selected_number_of_frames_per_video==self.original_number_of_frames_per_video else masks_read['masks'][0::self.step,:,:]
        rgbs_read_torch=torch.stack(rgbs_read)
        #flows_read_torch=torch.stack(flows_read)

        return bbox_read_torch,masks_read_torch,rgbs_read_torch
    
    def skipframes(self,rgbs):
        output=[]
        for i in range (0,len(self.rgb_addresses),self.step):
            output.append(rgbs[i])   
        return output