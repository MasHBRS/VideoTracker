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

class MultiHeadSelfAttention(nn.Module):
    """ 
    Self-Attention module

    Args:
    -----
    token_dim: int
        Dimensionality of the tokens in the transformer
    inner_dim: int
        Dimensionality used for attention
    """

    def __init__(self, token_dim, attn_dim, num_heads):
        """ """
        super().__init__()
        self.token_dim = token_dim #  Embedding size per token, here called D. I believe this is the x (input) dimension.
        self.attn_dim = attn_dim # the dimension of the attention vector, and will be splitted among heads.
        self.num_heads = num_heads 
        assert num_heads >= 1 # multi-head attention
        assert attn_dim % num_heads == 0, f"attn_dim = {attn_dim} must be divisible by num_heads = {num_heads}..."
        self.head_dim = attn_dim // num_heads

        # query, key and value projections
        self.q = nn.Linear(token_dim, attn_dim, bias=False) 
        self.k = nn.Linear(token_dim, attn_dim, bias=False) 
        self.v = nn.Linear(token_dim, attn_dim, bias=False) 

        # output projection
        self.out_proj = nn.Linear(attn_dim, token_dim, bias=False) # back to the original input dimension
        return
    
    def attention(self, query, key, value):
        """
        Computing self-attention

        All (q,k,v) ~ (B, N, D)
        """
        scale = (query.shape[-1]) ** (-0.5) # smoothing gradiants to work better with softmax

        # similarity between each query and the keys
        similarity = torch.bmm(query, key.permute(0, 2, 1)) * scale  # ~(B, N, N) batch-wise matrix multiplication, permmute here acts as traspose for dimentions matching
        attention = similarity.softmax(dim=-1) # softmax across each row 
        self.attention_map = attention # for visualization \latter

        # attention * values
        output = torch.bmm(attention, value)
        return output

    def split_into_heads(self, x):
        """
        Splitting a vector into multiple heads
        """
        batch_size, num_tokens, _ = x.shape # (2, 5, 64)
        x = x.view(batch_size, num_tokens, self.num_heads, self.head_dim)  # split dimension into heads (2, 5, 8, 8)
        y = x.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, num_tokens, self.head_dim)  
        # (2, 8, 5, 8) --> (16, 5, 8)
        # permute: This allows each head to attend independently to all tokens in the sequence.
        # (batch_size, num_heads, num_tokens, head_dim)
        # reshape: This combines the heads into a single dimension, allowing for batch-wise operations.

        return y

    def merge_heads(self, x):
        """
        Rearranging heads and recovering original shape
        """
        _, num_tokens, dim_head = x.shape # (16, 5, 8)
        x = x.reshape(-1, self.num_heads, num_tokens, dim_head).transpose(1, 2) # (2, 8, 5, 8) --> (2, 5, 8, 8)
        y = x.reshape(-1, num_tokens, self.num_heads * dim_head) # (2, 5, 64)
        return y


    def forward(self, x):
        """ 
        Forward pass through Self-Attention module
        """
        # linear projections and splitting into heads:
        # (B, N, D) --> (B, N, Nh, Dh) --> (B * Nh, N, Dh)
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        # applying attention equation
        vect = self.attention(query=q, key=k, value=v)

        # rearranging heads and recovering shape:
        # (B * Nh, N, Dh) --> (B N, Nh, Dh) --> (B, N, D) same shape as the input
        y = self.merge_heads(vect)

        y = self.out_proj(y) #(B, N, token_dim)
        return y

class MLP(nn.Module):
    """
    2-Layer Multi-Layer Perceptron used in transformer blocks
    
    Args:
    -----
    in_dim: int
        Dimensionality of the input embeddings to the MLP
    hidden_dim: int
        Hidden dimensionality of the MLP
    """
    
    def __init__(self, in_dim, hidden_dim):
        """ MLP Initializer """
        super().__init__()
        self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),  ## NOTE: GELU activation function used in FCL for transformers!
                nn.Linear(hidden_dim, in_dim),
            )
        
    def forward(self, x):
        """ Forward """
        y = self.mlp(x)
        return y

class TransformerBlock(nn.Module):
    """
    Transformer block using self-attention

    Args:
    -----
    token_dim: int
        Dimensionality of the input tokens
    attn_dim: int
        Inner dimensionality of the attention module. Must be divisible be num_heads
    num_heads: int
        Number of heads in the self-attention mechanism
    mlp_size: int
        Hidden dimension of the MLP module
    """

    def __init__(self, token_dim, attn_dim, num_heads, mlp_size):
        """ Module initializer """
        super().__init__()
        self.token_dim = token_dim
        self.mlp_size = mlp_size
        self.attn_dim = attn_dim
        self.num_heads = num_heads

        # MHA
        self.ln_att = nn.LayerNorm(token_dim, eps=1e-6) # Layer normalization
        self.attn = MultiHeadSelfAttention(
                token_dim=token_dim,
                attn_dim=attn_dim,
                num_heads=num_heads
            )
        
        # MLP
        self.ln_mlp = nn.LayerNorm(token_dim, eps=1e-6) # Layer normalization
        self.mlp = MLP(
                in_dim=token_dim,
                hidden_dim=mlp_size,
            )
        return


    def forward(self, inputs):
        """
        Forward pass through transformer encoder block.
        We assume the more modern PreNorm design
        """
        assert inputs.ndim == 3 # (B, N, D)

        # Self-attention.
        x = self.ln_att(inputs)
        x = self.attn(x)
        y = x + inputs # residual connection

        # MLP
        z = self.ln_mlp(y)
        z = self.mlp(z)
        z = z + y # residual connection

        return z


    def get_attention_masks(self):
        """ Fetching last computer attention masks """
        attn_masks = self.attn.attention_map
        N = attn_masks.shape[-1]
        attn_masks = attn_masks.reshape(-1, self.num_heads, N, N)
        return attn_masks

class Patchifier:
    """ 
    Module that splits an image into patches.
    We assumen square images and patches
    """

    def __init__(self, patch_size):
        """ """
        self.patch_size = patch_size

    def __call__(self, img):
        """ """
        B, C, H, W = img.shape
        assert H % self.patch_size == 0
        assert W % self.patch_size == 0
        num_patch_H = H // self.patch_size
        num_patch_W = W // self.patch_size

        # splitting and reshaping
        patch_data = img.reshape(B, C, num_patch_H, self.patch_size, num_patch_W, self.patch_size)
        patch_data = patch_data.permute(0, 2, 4, 1, 3, 5)  # ~(B, n_p_h, n_p_w, C, p_s, p_s)
        patch_data = patch_data.reshape(B, num_patch_H * num_patch_W, C * self.patch_size * self.patch_size)
        return patch_data

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional encoding 

    Args:
    -----
    d_model: int
        Dimensionality of the slots/tokens
    max_len: int
        Length of the sequence.
    """

    def __init__(self, d_model, max_len=50):
        """
        Initializing the positional encoding
        """
        super().__init__()
        self.d_model = d_model #  The dimensionality of token embeddings
        self.max_len = max_len #  Maximum sequence length the model can handle (default 50)

        # initializing embedding
        self.pe = self._get_pe()
        return

    def _get_pe(self):
        """
        Initializing the temporal positional encoding given the encoding mode
        """
        max_len = self.max_len
        d_model = self.d_model
        
        pe = torch.zeros(max_len, d_model) # Creates a zero tensor - one row per position
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # Even dimensions get sine
        pe[:, 1::2] = torch.cos(position * div_term) # Odd dimensions get cosine
        pe = pe.view(1, max_len, d_model)
        return pe

    def forward(self, x):
        """
        Adding the positional encoding to the input tokens of the transformer
        """
        if x.device != self.pe.device:
            self.pe = self.pe.to(x.device)
        batch_size, num_tokens = x.shape[0], x.shape[1]   # What?? X has shape of [B, number of tokens, dim]
         # Repeat for batch and truncate to actual sequence length
        cur_pe = self.pe.repeat(batch_size, 1, 1)[:, :num_tokens] # what is happeninig here? second dim refers to the number of tokens, and has a max value on the constructor. So we return the first 'num_token' values of the list.  

        y = x + cur_pe # Adding the positional encoding to the input tokens
        return y