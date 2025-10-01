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
from models.transformers.attention_mechanism import Patchifier,PositionalEncoding,EncoderBlock
from utils.util import ImageExtractor


class ViT_ImageBased(nn.Module):
    """ 
    Vision Transformer for image classification
    """

    def __init__(self,img_size, frame_numbers, patch_size, token_dim, attn_dim, num_heads, mlp_size, num_tf_layers):
        """ Model initializer
         num_tf_layers : The number of transformer blocks that we need
           """
        super().__init__()

        # breaking image into patches, and projection to transformer token dimension
        self.pathchifier = Patchifier(patch_size) ##(BatchSize, Frames, num_patch_H * num_patch_W, C * self.patch_size * self.patch_size)

        ''' Creating the embedding for each image patch/token'''
        self.patch_projection = nn.Sequential(   
                nn.LayerNorm(patch_size * patch_size * 3),
                nn.Linear(patch_size * patch_size * 3, token_dim) # token_dim = token embedding
            )
        patch_counts=(img_size//patch_size)**2
        self.cls_token = nn.Parameter(torch.randn(1, token_dim) / (token_dim ** 0.5), requires_grad=True)
        self.pos_emb = PositionalEncoding(token_dim,max_len=frame_numbers)# return token embeddings + positional encoding

        # cascade of transformer blocks
        transformer_blocks = [
            EncoderBlock(
                    token_dim=token_dim,
                    attn_dim=attn_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size
                )
            for _ in range(num_tf_layers)
        ]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        return

    
    def forward(self, x,boxes=None,masks=None): # full Transformer encoder block forward pass
        """ 
        Forward pass
        """
        B,frameNumbers, channel,img_height,img_width = x.shape  # (B,frameNumbers, channel,img_height,img_width)
        
        # breaking image into patches, and projection to transformer token dimension
        patches = self.pathchifier(x)  # (B, frameNumbers, PatchCounts, PatchSize * PatchSize * 3)
        patch_tokens = self.patch_projection(patches)  # (B, frameNumbers, PatchCounts, token_dim)

        # concatenating CLS token and adding positional embeddings
        cur_cls_token = self.cls_token.unsqueeze(0).unsqueeze(0).repeat(B, frameNumbers,1, 1) # shape: (B,frameNumber,1, D).  B copies of the token, one for each sample in the batch.
        tokens = torch.cat([cur_cls_token, patch_tokens], dim=2)  # ~(B, 1 + 16, D) So now, each input sequence starts with the [CLS] token, followed by the patch tokens.
        # ---> one token for all the patches of one image, not per batch. Summary token of all image
        tokens_with_pe = self.pos_emb(tokens)

        # processing with transformer
        out_tokens = self.transformer_blocks(tokens_with_pe) #  (B,frame_numbers, max_objects_in_scene + 1 , token_dim)
        out_cls_token = out_tokens[:,:, 0]  # fetching only CLS token

        return out_cls_token #(B,frame_numbers, token_dim)


    def get_attn_mask(self):
        """
        Fetching the last attention maps from all TF Blocks
        """
        attn_masks = [tf.get_attention_masks() for tf in self.transformer_blocks]
        return attn_masks
