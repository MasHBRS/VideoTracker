import shutil
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'src')))
from loader.Dataset import VideoDataset
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import io
import glob
from torch.utils.data import DataLoader

class VideoDataset_ImageLevel(Dataset):
    def __init__(self, data_directory, frames_per_video=24, sample_frames=10, transform=None):
        """
        frame_paths: list of all frame file paths in order
        frames_per_video: total frames per video (e.g. 24)
        sample_frames: number of frames to sample per video (e.g. 10)
        transform: torchvision transforms to apply to frames
        """
        self.frame_paths=sorted(glob.glob(os.path.join(data_directory,'rgb*.png')))
        self.frames_per_video = frames_per_video
        self.sample_frames = sample_frames
        self.transform = transform

        # Number of videos
        self.num_videos = len(self.frame_paths) // frames_per_video

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx):
        # Get frames for this video
        start = idx * self.frames_per_video
        end = start + self.frames_per_video
        video_frames = self.frame_paths[start:end]

        # Uniformly spaced frame indices
        indices = [int(i * (self.frames_per_video - 1) / (self.sample_frames - 1))
                   for i in range(self.sample_frames)]
        selected_frames = [video_frames[i] for i in indices]

        # Load and transform frames
        frames = []
        for f in selected_frames:
            img = Image.open(f).convert("RGB")
            if self.transform:
                img = self.transform(img)
            else:
                img = T.ToTensor()(img)
            frames.append(img)

        # Stack -> [frames, C, H, W]
        frames = torch.stack(frames, dim=0)
        return frames   # shape: [10, C, H, W]

# Example usage:
"""transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])
"""
transform=None
# Suppose `frames` is your list of file paths
dataset = VideoDataset_ImageLevel('/home/nfs/inf6/data/datasets/MOVi/movi_c/', frames_per_video=24, sample_frames=10, transform=transform)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Get a batch
batch = next(iter(dataloader))
print(batch.shape)  # [8, 10, 3, 128, 128]
