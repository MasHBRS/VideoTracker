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
from models.transformers.self_attention import Patchifier,PositionalEncoding,TransformerBlock
from utils.util import ImageExtractor

class ViT(nn.Module):
    """ 
    Vision Transformer for image classification
    """

    def __init__(self, patch_size, token_dim, attn_dim, num_heads, mlp_size, num_tf_layers, num_classes):
        """ Model initializer
         num_tf_layers : The number of transformer blocks that we need
           """
        super().__init__()

        # breaking image into patches, and projection to transformer token dimension
        self.pathchifier = Patchifier(patch_size)

        ''' Creating the embedding for each image patch/token'''
        self.patch_projection = nn.Sequential(   
                nn.LayerNorm(patch_size * patch_size * 3),
                nn.Linear(patch_size * patch_size * 3, token_dim) # token_dim = token embedding
            )

        # adding CLS token and positional embedding
        '''nn.Parameter: To assign a tensor as Module attributes they are automatically added to the list of
        its parameters, and will appear e.g. in ~Module.parameters iterator. when requires_grad = True ---> the tensor
        is updated through training with GD. 
        / (token_dim ** 0.5): a common trick to stabilize training by keeping the scale of weights roughly controlled (similar to Xavier initialization)'''
        self.cls_token = nn.Parameter(torch.randn(1, token_dim) / (token_dim ** 0.5), requires_grad=True)
        self.pos_emb = PositionalEncoding(token_dim) # return token embeddings + positional encoding

        # cascade of transformer blocks
        transformer_blocks = [
            TransformerBlock(
                    token_dim=token_dim,
                    attn_dim=attn_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size
                )
            for _ in range(num_tf_layers)
        ]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)

        # classifier
        self.classifier = nn.Linear(token_dim, num_classes)
        self.extractor=ImageExtractor()
        return

    def apply_masks_or_bboxes(self,x,apply_mask=None,apply_bbox=None,max_objects_in_scene=0):
        output=[]
        assert x.dim()==5
        B,frame_numbers,channels,height,width = x.shape
        for batchIndx in range(x.shape[0]):
            for frameIndx in range(x.shape[1]):
                if apply_mask is not None:
                    output.append(self.extractor.extract_masked_images(x[batchIndx,frameIndx],masks=apply_mask[batchIndx,frameIndx],max_objects_in_scene=max_objects_in_scene)[0])
                elif apply_bbox is not None:
                    output.append(self.extractor.extract_bboxed_images(x[batchIndx,frameIndx],bboxes=apply_bbox[batchIndx,frameIndx],max_objects_in_scene=max_objects_in_scene))
        outputTorch=torch.stack(output,dim=0).reshape(B,frame_numbers,max_objects_in_scene,channels,height,width)
        return outputTorch

    def forward(self, x, boxes=None,masks=None,max_objects_in_scene=0): # full Transformer encoder block forward pass
        """ 
        Forward pass
        """
        assert (boxes==None) ^ (masks==None), "One of boxes or masks should be valued, and not both!" # either one of them should be non-empty.
        assert x.dim()==5
        B,frame_numbers,channels,height,width = x.shape
        

        self.apply_masks_or_bboxes(x,apply_mask=masks,apply_bbox=boxes,max_objects_in_scene=max_objects_in_scene)
        # breaking image into patches, and projection to transformer token dimension
        patches = self.pathchifier(x)  # (B, 16, 8 * 8 * 3)
        patch_tokens = self.patch_projection(patches)  # (B, 16, D)

        # concatenating CLS token and adding positional embeddings
        cur_cls_token = self.cls_token.unsqueeze(0).repeat(B, 1, 1) # shape: (B, 1, D).  B copies of the token, one for each sample in the batch.
        tokens = torch.cat([cur_cls_token, patch_tokens], dim=1)  # ~(B, 1 + 16, D) So now, each input sequence starts with the [CLS] token, followed by the patch tokens.
        # ---> one token for all the patches of one image, not per batch. Summary token of all image
        tokens_with_pe = self.pos_emb(tokens)

        # processing with transformer
        out_tokens = self.transformer_blocks(tokens_with_pe) # shapes?
        out_cls_token = out_tokens[:, 0]  # fetching only CLS token

        # classification
        logits = self.classifier(out_cls_token)
        return logits


    def get_attn_mask(self):
        """
        Fetching the last attention maps from all TF Blocks
        """
        attn_masks = [tf.get_attention_masks() for tf in self.transformer_blocks]
        return attn_masks