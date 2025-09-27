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
from models.transformers.attention_mechanism import Patchifier,PositionalEncoding,EncoderBlock
from utils.util import ImageExtractor

class ViT(nn.Module):
    """ 
    Vision Transformer for image classification
    """

    def __init__(self, img_height,img_width,channels,frame_numbers, token_dim, attn_dim, num_heads, mlp_size, num_tf_layers,max_objects_in_scene):#, num_classes):
        """ Model initializer
         num_tf_layers : The number of transformer blocks that we need
           """
        super().__init__()

        self.max_objects_in_scene=max_objects_in_scene

        self.patch_projection = nn.Sequential(   
                nn.LayerNorm(img_height * img_width * channels),
                nn.Linear(img_height * img_width * channels, token_dim) # token_dim = token embedding
            )

        self.cls_token = nn.Parameter(torch.randn(1, token_dim) / (token_dim ** 0.5), requires_grad=True)
        self.pos_emb = PositionalEncoding(token_dim,max_len=frame_numbers) # return token embeddings + positional encoding

        encoderBlocks = [
            EncoderBlock(
                    token_dim=token_dim,
                    attn_dim=attn_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size
                )
            for _ in range(num_tf_layers)
        ]
        self.encoderBlocks = nn.Sequential(*encoderBlocks)

        self.extractor=ImageExtractor(max_objects_in_scene)
        return

    def apply_masks_or_bboxes(self,x,apply_mask=None,apply_bbox=None):
        output=[]
        assert x.dim()==5
        B,frame_numbers,channels,height,width = x.shape
        for batchIndx in range(x.shape[0]):
            for frameIndx in range(x.shape[1]):
                if apply_mask is not None:
                    output.append(self.extractor.extract_masked_images(x[batchIndx,frameIndx],masks=apply_mask[batchIndx,frameIndx])[0])
                elif apply_bbox is not None:
                    output.append(self.extractor.extract_bboxed_images(x[batchIndx,frameIndx],bboxes=apply_bbox[batchIndx,frameIndx]))
        outputTorch=torch.stack(output,dim=0).to(x.device).reshape(B,frame_numbers,self.max_objects_in_scene,channels,height,width)
        return outputTorch

    def forward(self, x, boxes=None,masks=None): # full Transformer encoder block forward pass
        """ 
        Forward pass
        """
        assert (boxes==None) ^ (masks==None), "One of boxes or masks should be valued, and not both!" # either one of them should be non-empty.
        assert x.dim()==5
        if (x.device!=next(self.parameters()).device):
            x=x.to(next(self.parameters()).device)
        if (masks is not None) and masks.device!=x.device:
            masks=masks.to(x.device)
        if (boxes is not None) and boxes.device!=x.device:
            boxes=boxes.to(x.device)
        B,frame_numbers,channels,height,width = x.shape
        
        filtered_imgs=self.apply_masks_or_bboxes(x,apply_mask=masks,apply_bbox=boxes) #(B,frame_numbers,max_objects_in_scene,channels,height,width)
        filtered_imgs=filtered_imgs.reshape(B,frame_numbers,self.max_objects_in_scene,-1)#(B,frame_numbers,max_objects_in_scene, channels * height * width)
        
        patch_tokens = self.patch_projection(filtered_imgs)  # (B,frame_numbers,max_objects_in_scene, token_dim)
        
        cur_cls_token = self.cls_token.unsqueeze(0).unsqueeze(0).repeat(B, frame_numbers,1, 1) # (B,frame_numbers, 1 , token_dim)

        tokens = torch.cat([cur_cls_token, patch_tokens], dim=2) # (B,frame_numbers, max_objects_in_scene + 1 , token_dim)

        tokens_with_pe = self.pos_emb(tokens)

        out_tokens = self.encoderBlocks(tokens_with_pe) # (B,frame_numbers, max_objects_in_scene + 1 , token_dim)
        out_cls_token = out_tokens[:,:, 0]  # fetching only CLS token

        return out_cls_token #(B, frame_numbers, token_dim )


    def get_attn_mask(self):
        """
        Fetching the last attention maps from all TF Blocks
        """
        attn_masks = [tf.get_attention_masks() for tf in self.encoderBlocks]
        return attn_masks