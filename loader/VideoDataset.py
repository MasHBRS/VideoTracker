import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms,io
from pathlib import Path
import glob
from collections import defaultdict

class VideoDataset(Dataset):
    def read_files(self,data_directory,condition):

        retrieved_addresses = glob.glob(os.path.join(data_directory,condition))
        if condition.endswith('png'):
            retrieved_files = [io.read_image(path=file).to(torch.float32) for file in retrieved_addresses]
        if condition.endswith('pt'):
            retrieved_files = [torch.load(file, map_location="cpu") for file in retrieved_addresses]
         
        return retrieved_files

    def __init__(self, data_path, split='train' ,transforms=None):
        data_directory=os.path.join(data_path, split)
        if not os.path.exists(data_directory):
            if not os.path.exists(os.path.abspath(data_directory)):
                raise Exception("Dataset was not found!")
        
        number_of_frames_per_video=24
        
        self.coord=self.read_files(data_directory,'coords*.pt')
        
        self.mask=self.read_files(data_directory,'mask*.pt')

        self.rgb=self.read_files(data_directory,'rgb*.png')
        self.rgb=self.group_by_film(self.rgb,group_size=number_of_frames_per_video)

        self.flow=self.read_files(data_directory,'flow*.png')
        self.flow=self.group_by_film(self.flow,group_size=number_of_frames_per_video)

        print(f"Data Loaded= {len(self.coord)=}, {len(self.mask)=}, {len(self.rgb)=}, {len(self.flow)=}")

    def group_by_film(self, files,group_size):
        groups = [torch.stack(files[i:i+group_size]) for i in range(0, len(files), group_size)]
        return groups

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, idx):
        return self.coord[idx],self.mask[idx],self.rgb[idx],self.flow[idx]
    
    def video_to_frames(self,com,bbox,masks, rgbs, flows):
        output=[]
        for i in range(com.shape[0]):
            output.append((com[i],bbox[i],masks[i],rgbs[i],flows[i]))
        
        return output # output, should have length=24. Each entry corresponds to data of a frame data.