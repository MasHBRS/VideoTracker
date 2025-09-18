import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import io
import glob

class VideoDataset(Dataset):
    
    def __init__(self, data_path, max_objects_in_scene, split='train',transforms=None,number_of_frames_per_video=24,device='cpu'):
        data_directory=os.path.join(data_path, split)
        if not os.path.exists(data_directory):
            if not os.path.exists(os.path.abspath(data_directory)):
                raise Exception("Dataset was not found!")
        self.number_of_frames_per_video=number_of_frames_per_video
        self.device=device
        self.max_objects_in_scene=max_objects_in_scene
        self.coord_addresses=sorted(glob.glob(os.path.join(data_directory,'coords*.pt')))
        self.mask_addresses=sorted(glob.glob(os.path.join(data_directory,'mask*.pt')))
        self.rgb_addresses=sorted(glob.glob(os.path.join(data_directory,'rgb*.png')))
        self.flow_addresses=sorted(glob.glob(os.path.join(data_directory,'flow*.png')))

        assert len(self.coord_addresses)==len(self.mask_addresses)==(len(self.rgb_addresses)/number_of_frames_per_video)==(len(self.flow_addresses)/number_of_frames_per_video)
        self.number_of_videos=len(self.rgb_addresses)/number_of_frames_per_video
        print(f"Data Loaded Successfully: {len(self.coord_addresses)=}, {len(self.mask_addresses)=}, {len(self.rgb_addresses)=}, {len(self.flow_addresses)=}")

    def __len__(self):
        return int(self.number_of_videos)

    def __getitem__(self, filmIndex):
        rgbs,flows=self.read_data(filmIndex)
        return rgbs,flows

    def read_data(self,filmIndex):
        coords_to_read=self.coord_addresses[filmIndex*self.number_of_frames_per_video:(filmIndex+1)*self.number_of_frames_per_video]
        masks_to_read=self.mask_addresses[filmIndex*self.number_of_frames_per_video:(filmIndex+1)*self.number_of_frames_per_video]
        rgbs_to_read=self.rgb_addresses[filmIndex*self.number_of_frames_per_video:(filmIndex+1)*self.number_of_frames_per_video]
        flows_to_read=self.flow_addresses[filmIndex*self.number_of_frames_per_video:(filmIndex+1)*self.number_of_frames_per_video]

        coords_read=[torch.load(file, map_location="cpu") for file in coords_to_read]
        coms_read=[com["com"] for com in coords_read]
        bbox_read=[box["bbox"] for box in coords_read]
        masks_read=[torch.load(file, map_location="cpu") for file in masks_to_read]
        masks_read_kyes=[mask["masks"] for mask in masks_read]
        rgbs_read=[io.read_image(path=file).to(torch.float32) for file in rgbs_to_read]
        flows_read=[io.read_image(path=file).to(torch.float32) for file in flows_to_read]

        coms_read_torch=torch.stack(coms_read) if len(coms_read)>0 else torch.zeros(size=(self.number_of_frames_per_video,self.max_objects_in_scene,2))
        #bboxs_read_torch=torch.stack(bbox_read)
        masks_read_torch=torch.stack(masks_read_kyes) if len(masks_read_kyes)>0 else torch.zeros(size=(self.number_of_frames_per_video,self.max_objects_in_scene,2))
        rgbs_read_torch=torch.stack(rgbs_read)
        flows_read_torch=torch.stack(flows_read)

        return coms_read_torch,rgbs_read_torch,flows_read_torch