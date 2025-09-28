import os
import random
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms

import sys
import os
sys.path.append(os.path.abspath("../.."))
from models.transformers.encoders.vit_encoder import ViT
from models.transformers.decoders.vit_decoder import ViT_Decoder 

class CustomizableTransformer(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
    def forward(self, x, boxes=None,masks=None):
        encoded=self.encoder(x, boxes,masks)
        normalized=self.normalize(encoded)
        decoded=self.normalize(self.decoder(normalized.unsqueeze(2)))
        return decoded
    def normalize(self,x):
        min,max=x.min(),x.max()
        norm=(x-min)/(max-min+1e-8)
        return norm