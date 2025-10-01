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
sys.path.append(os.path.abspath("../../../.."))
from models.transformers.attention_mechanism import PositionalEncoding,DecoderBlock

class ViT_Decoder(nn.Module):
    def __init__(self, batch_size, img_height,img_width,channels,frame_numbers, token_dim, attn_dim, num_heads, mlp_size, num_tf_layers,device,max_objects_in_scene=0):
        super().__init__()
        self.token_dim=token_dim
        self.batch_size, self.img_height,self.img_width,self.channels,self.frame_numbers,self.max_objects_in_scene=batch_size, img_height,img_width,channels,frame_numbers,max_objects_in_scene
        """self.patch_projection = nn.Sequential(   
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim,img_height * img_width * channels ) # token_dim = token embedding
            ).to(device)"""
        self.patch_projection= nn.Sequential(
            # [1, token_dim, 1, 1] -> [1, 256, 4, 4]
            nn.ConvTranspose2d(token_dim, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # [1, 256, 4, 4] -> [1, 128, 8, 8]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # [1, 128, 8, 8] -> [1, 64, 16, 16]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # [1, 64, 16, 16] -> [1, 32, 32, 32]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # [1, 32, 32, 32] -> [1, channels, output_height, output_width]
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # or nn.Sigmoid() depending on your data range
        )
        self.pos_emb = PositionalEncoding(token_dim,max_len=frame_numbers).to(device) # return token embeddings + positional encoding

        # cascade of transformer blocks
        self.decoderBlocks = [
            DecoderBlock(
                    token_dim=token_dim,
                    attn_dim=attn_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size
                ).to(device)
            for _ in range(num_tf_layers)
        ]
    
        self.query_shifted_right=nn.Parameter(torch.randn(batch_size, frame_numbers, 1, token_dim)) # This 1 is for resembeling encoder input shape.
        return

    def forward(self, encoder_output): # full Transformer decoder block forward pass
        """ 
        Forward pass
        """
        query=self.query_shifted_right
        query = self.pos_emb(query)
        
        output_decoder_blocks=query
        for block in self.decoderBlocks:
            output_decoder_blocks=block(output_decoder_blocks,encoder_output)
        output_decoder_blocks=output_decoder_blocks.reshape(-1,self.token_dim,1,1)# [Batch_size * number_of_frames, token_dim, 1, 1] 
        patch_tokens = self.patch_projection(output_decoder_blocks)
        patch_tokens=patch_tokens.reshape(self.batch_size,self.frame_numbers,self.channels,self.img_height,self.img_width)
        return patch_tokens


    def get_attn_mask(self):
        """
        Fetching the last attention maps from all TF Blocks
        """
        attn_masks = [tf.get_attention_masks() for tf in self.decoderBlocks]
        return attn_masks