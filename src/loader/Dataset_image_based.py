import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import io
import glob

class VideoDataset_ImageBased(Dataset):
    
    def __init__(self, data_path, halve_dataset,is_test_dataset, selected_number_of_frames_per_video=24,original_number_of_frames_per_video=24, split='train',transforms=None,device='cpu'):
        data_directory=os.path.join(data_path, split)
        if not os.path.exists(data_directory):
            if not os.path.exists(os.path.abspath(data_directory)):
                raise Exception("Dataset was not found!")
        self.selected_number_of_frames_per_video=selected_number_of_frames_per_video
        self.original_number_of_frames_per_video=original_number_of_frames_per_video
        self.device=device
        self.transforms=transforms
        self.rgb_addresses=sorted(glob.glob(os.path.join(data_directory,'rgb*.png')))
        if(selected_number_of_frames_per_video!=original_number_of_frames_per_video):
            self.step=original_number_of_frames_per_video//selected_number_of_frames_per_video
            self.rgb_addresses=self.skipframes(rgbs=self.rgb_addresses)
        self.halve_dataset=halve_dataset
        self.is_test_dataset=is_test_dataset
        self.number_of_videos=len(self.rgb_addresses)//selected_number_of_frames_per_video

    def __len__(self):
        return self.number_of_videos if self.halve_dataset==False else  self.number_of_videos//2

    def __getitem__(self, filmIndex): 
        if self.halve_dataset and self.is_test_dataset:
                filmIndex+=self.__len__()
        rgbs=self.read_data(filmIndex)
        if self.transforms!=None:
            rgbs = self.transforms(rgbs)
        return rgbs

    def read_data(self,filmIndex):
        rgbs_to_read=self.rgb_addresses[filmIndex*self.selected_number_of_frames_per_video:(filmIndex+1)*self.selected_number_of_frames_per_video]
        rgbs_read=[io.read_image(path=file).to(torch.float32) for file in rgbs_to_read]
        rgbs_read_torch=torch.stack(rgbs_read)
        return rgbs_read_torch
    
    def skipframes(self,rgbs):
        output=[]
        for i in range (0,len(self.rgb_addresses),1):
            temp_rgbs=self.rgb_addresses[i*self.original_number_of_frames_per_video:(i+1)*self.original_number_of_frames_per_video]
            temp_rgbs=temp_rgbs[0:self.step*self.selected_number_of_frames_per_video:self.step]
            output.extend(temp_rgbs)
        return output