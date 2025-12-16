# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from vggt.utils.load_fn import load_and_preprocess_images
import numpy as np
from vggt.layers.ot_matcher import SoftmaxMatcher
from vggt.layers.attention_knockout import *
from vggt.layers.token_weighter import TokenFusionStrategy
from merging.merge import token_merge_bipartite2d

XFORMERS_AVAILABLE = False

def print_tensor_info(tensor, name="Tensor"):
    """
    打印张量的名字、形状和占用的显存大小（单位MB），以及所在设备。
    """
    if isinstance(tensor, torch.Tensor):
        num_bytes = tensor.element_size() * tensor.numel()
        mem_mb = num_bytes / 1024 / 1024
        print(f"{name}: shape={tuple(tensor.shape)}, device={tensor.device}, size={mem_mb:.2f} MB")
    else:
        print(f"{name}: 不是torch.Tensor类型，实际类型: {type(tensor)}")


class GlobalAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        token_weighter: TokenFusionStrategy = None,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # for zcl test
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        self.ot_params = {
            'reg':0.1,
            'reg_kl':10,
            'sinkhorn_iterations':10,
            'mass':0.9,
            'bin_score':0.3
        }
        self.matcher = SoftmaxMatcher(**self.ot_params)
        self.matcher.eval()
        self.token_weighter = token_weighter
    
    def set_use_fused_attn(self, use_fused: bool) -> None:
        print("set_use_fused_attn: ", use_fused)
        self.fused_attn = use_fused

    def forward(self, x: Tensor, pos=None, idx=None) -> Tensor:

        # skip_layer_idx = [0, 1, 2, 3]
        # if idx in skip_layer_idx:
        #     return x

        self.token_weighter.inspect_token_similarity(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k) # (B, num_heads, N, head_dim)
        q, k = self.token_weighter.rope(self.rope, q, k, pos)
        q, k, v =self.token_weighter.token_merge_FastVGGT(x, q, k, v)
        N_m = q.shape[-2]
        if N_m != N:
            print(f"token merge ratio of layer {idx}: {N_m / N}")

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)

        # # if idx == 15:
        # #     q = q[:, 15, :, :]
        # #     k = k[:, 15, :, :]
        # #     v = v[:, 15, :, :]

        # q = q * self.scale
        # attn_map = q @ k.transpose(-2, -1)
        # attn_map = attn_map.softmax(dim=-1)
        # # if idx in range(12, 16, 1):
        # # if idx == 15:
        # #     attn_map = attn_map[:, 15, :, :]
        # #     attn_map = attn_map.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        # # if idx == 17:
        # #     attn_map = attn_map[:, 5, :, :]
        # #     attn_map = attn_map.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        # attn_map = self.token_weighter.attention_knockout(attn_map, idx)
        # self.token_weighter.visualize_attn_map_all_heads(attn_map, 491+5, 1)
        # self.token_weighter.calculate_top_k_dominance(attn_map)
        # x = self.attn_drop(attn_map) @ v
        # # if idx == 15:
        # #     x = x.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # del v, attn_map

        x = x.transpose(1, 2).reshape(B, N_m, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.token_weighter.token_unmerge_FastVGGT(x)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # for zcl test
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
    
    def set_use_fused_attn(self, use_fused: bool) -> None:
        print("set_use_fused_attn: ", use_fused)
        self.fused_attn = use_fused

    def forward(self, x: Tensor, pos=None, idx=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        attn_before_softmax = None

        # if self.fused_attn:
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        # else:
        #     q = q * self.scale
        #     attn_map = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        #     attn = self.attn_drop(attn_map)
        #     x = attn @ v
        #     del attn, attn_map
            
        del q, k, v
            
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, idx=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x