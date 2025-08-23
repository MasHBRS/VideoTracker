import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,io

class VideoLoader(Dataset):
    def __init__(self, data_path, transforms=None):
        self.masks=[]
    

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        return self.masks[idx]