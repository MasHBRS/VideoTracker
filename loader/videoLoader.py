import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,io
from pathlib import Path
import glob

class VideoLoader(Dataset):
    def read_files(self,data_directory,condition):
        print(data_directory, condition)
        retrieved_addresses = glob.glob(os.path.join(data_directory,condition))
        if condition.endswith('png'):
            retrieved_files = [io.read_image(path=file).to(torch.float32) for file in retrieved_addresses]
        if condition.endswith('pt'):
            retrieved_files = [torch.load(file, map_location="cpu") for file in retrieved_addresses]
         
        print(condition,len(retrieved_files))
        return retrieved_files

    def __init__(self, data_path, split='train' ,transforms=None):
        data_directory=os.path.join(data_path, split)
        if not os.path.exists(data_directory):
            if not os.path.exists(os.path.abspath(data_directory)):
                raise Exception("Dataset was not found!")
        
        self.rgb=self.read_files(data_directory,'rgb*.png')
        self.mask=self.read_files(data_directory,'mask*.pt')
        self.flow=self.read_files(data_directory,'flow*.png')
        self.coord=self.read_files(data_directory,'coords*.pt')

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, idx):
        return self.mask[idx]