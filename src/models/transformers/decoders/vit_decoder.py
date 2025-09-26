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
sys.path.append(os.path.abspath("../../.."))
from models.transformers.attention_mechanism import Patchifier,PositionalEncoding,DecoderBlock

class ViT_Decoder(nn.Module):
    def __init__(self, batch_size, img_height,img_width,channels,frame_numbers, token_dim, attn_dim, num_heads, mlp_size, num_tf_layers,max_objects_in_scene,device):
        super().__init__()
        self.batch_size, self.img_height,self.img_width,self.channels,self.frame_numbers,self.max_objects_in_scene=batch_size, img_height,img_width,channels,frame_numbers,max_objects_in_scene
        self.patch_projection = nn.Sequential(   
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim,img_height * img_width * channels ) # token_dim = token embedding
            ).to(device)
        """
        TODO:
        The parameters for PositionalEncoding, should be reconsidered.  
        """
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
        #self.decoderBlocks = nn.Sequential(*decoderBlocks)
        
        """ TODO: I have to modify the initial shape of self.query_shifter_right same as the shape of encoder_output  
        """
        self.query_shifted_right=nn.Parameter(torch.randn(batch_size,frame_numbers,max_objects_in_scene, token_dim))
        self.output_projector=nn.Sequential(
            nn.Conv2d(self.max_objects_in_scene * self.channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, channels, kernel_size=3, padding=1)
        ).to(device)
        return

    def forward(self, encoder_output): # full Transformer encoder block forward pass
        """ 
        Forward pass
        """
        query=self.query_shifted_right
        query = self.pos_emb(query)
        
        output_decoder_blocks=query
        for block in self.decoderBlocks:
            output_decoder_blocks=block(output_decoder_blocks,encoder_output)
        
        patch_tokens = self.patch_projection(output_decoder_blocks) 
        #patch_tokens=patch_tokens.softmax(dim=-1)
        patch_tokens=patch_tokens.reshape(-1,self.max_objects_in_scene*self.channels,self.img_height,self.img_width)
        patch_tokens=self.output_projector(patch_tokens).reshape(self.batch_size,self.frame_numbers,self.channels,self.img_height,self.img_width)
        return patch_tokens


    def get_attn_mask(self):
        """
        Fetching the last attention maps from all TF Blocks
        """
        attn_masks = [tf.get_attention_masks() for tf in self.encoderBlocks]
        return attn_masks