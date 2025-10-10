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

from torch import Tensor
from torch import nn
import torch.nn.functional as F
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from vggt.utils.load_fn import load_and_preprocess_images
import numpy as np

XFORMERS_AVAILABLE = False

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
        output_attn_map: bool = False,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # for zcl test
        self.fused_attn = fused_attn
        self.output_attn_map = output_attn_map

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        
        self.image_names = []
        for i in range(10):
            self.image_names += ["/home/rokae/zcl/vggt/examples/kitchen/images/{:02d}.png".format(i)]
        self.images = load_and_preprocess_images(self.image_names)
    
    def set_use_fused_attn(self, use_fused: bool) -> None:
        print("set_use_fused_attn: ", use_fused)
        self.fused_attn = use_fused

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        attn_before_softmax = None

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn_before_softmax = q @ k.transpose(-2, -1)
            attn_after_softmax = attn_before_softmax.softmax(dim=-1)
            attn = self.attn_drop(attn_after_softmax)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.output_attn_map:
            return x, attn_before_softmax
        else:
            return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
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

def resize_attention_map(attn_map, target_shape):
    attn_map_normalized = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    attn_map_resized = cv.resize(attn_map_normalized, (target_shape[1], target_shape[0]), interpolation=cv.INTER_LINEAR)
    return attn_map_resized

def apply_heatmap(image, heatmap, alpha=0.6):
    """
    :param image: 原始图像，shape (H, W, 3)
    :param heatmap: 注意力图，shape (H, W)，值范围 [0, 1]
    :param alpha: 热图透明度
    """
    # 确保图像数据类型一致
    image *= 255
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    heatmap_colored = cv.applyColorMap(np.uint8(255 * heatmap), cv.COLORMAP_JET)
    
    # 如果图像不是 uint8 类型，进行转换
    if heatmap_colored.dtype != np.uint8:
        heatmap_colored = heatmap_colored.astype(np.uint8)
    
    
    # image_enhanced = cv.convertScaleAbs(image, alpha=1.5, beta=0)
    output = cv.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return output