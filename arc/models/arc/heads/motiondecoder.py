# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from arc.models.arc.layers.block import Block, CrossBlock, AdaLNBlock
from arc.models.arc.layers.attention import CrossAttention
from arc.models.arc.layers.rope import RotaryPositionEmbedding2D, PositionGetter

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class MotionDecoder(nn.Module):
    def __init__(
        self,
        patch_size=14,
        embed_dim=1024,
        depth=4,
        num_heads=16,
        mlp_ratio=4.0,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        use_adaln=False,
        has_self_attention=True,
        has_cross_attention=True,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.use_adaln = use_adaln
        self.has_self_attention = has_self_attention
        self.has_cross_attention = has_cross_attention

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        if self.has_cross_attention:
            self.cross_blocks = nn.ModuleList(
                [
                    CrossBlock(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        cross_attn_init_values=init_values,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope,
                        attn_class=CrossAttention,
                    )
                    for _ in range(depth)
                ]
            )

        if self.has_self_attention:
            self.self_blocks = nn.ModuleList(
                [
                    (AdaLNBlock if use_adaln else block_fn)(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope,
                    )
                    for _ in range(depth)
                ]
            )

        self.depth = depth

    def forward(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor,
        patch_start_idx: int,
        track_query_idx = 0,
    ) -> torch.Tensor:
        """
        Args:
            tokens: [B, S, N, C], where N = 1 + 1 + 4 + P
            patch_start_idx: index where patches start
        """
        B, S, _, C = tokens.shape
        _, _, _, H, W = images.shape

        patches = tokens[:, :, patch_start_idx:, :] # [B, S, P, C]
        P = patches.shape[2]

        query_patches = patches[:, track_query_idx:track_query_idx+1, :, :]
        query_patches = query_patches.expand(B, S, P, C)
        
        time_emb = tokens[:, :, 1:2, :]
        
        time_cond = None
        if self.use_adaln:
            time_cond = time_emb.flatten(0, 2)
        
        # Concat time token to query patches
        query = torch.cat([time_emb, query_patches], dim=2) # [B, S, 1+P, C]
        
        kv = patches
        
        # 3. Prepare Positional Embeddings
        pos_q = None
        pos_k = None
        
        if self.position_getter is not None:
            pos_patches = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

            pos_patches = pos_patches + 1

            pos_time = torch.zeros(B * S, 1, 2, device=tokens.device, dtype=pos_patches.dtype)

            pos_q = torch.cat([pos_time, pos_patches], dim=1)

            pos_k = pos_patches

            pos_cross = (pos_q, pos_k)

        query = query.flatten(0, 1)
        kv = kv.flatten(0, 1)

        for cur_i in range(self.depth):
            # Cross Attention
            if self.has_cross_attention:
                if cur_i > 1 and self.training:
                    query = checkpoint(
                        lambda q, k, p: self.cross_blocks[cur_i](q, k, pos=p),
                        query, kv, pos_cross,
                        use_reentrant=False
                    )
                else:
                    query = self.cross_blocks[cur_i](query, kv, pos=pos_cross)

            if self.has_self_attention:
                # Self Attention
                if cur_i > 1 and self.training:
                    if self.use_adaln:
                        query = checkpoint(
                            lambda q, c, p: self.self_blocks[cur_i](q, cond=c, pos=p),
                            query, time_cond, pos_q,
                            use_reentrant=False
                        )
                    else:
                        query = checkpoint(
                            lambda q, p: self.self_blocks[cur_i](q, pos=p),
                            query, pos_q,
                            use_reentrant=False
                        )
                else:
                    if self.use_adaln:
                        query = self.self_blocks[cur_i](query, cond=time_cond, pos=pos_q)
                    else:
                        query = self.self_blocks[cur_i](query, pos=pos_q)

        query = query.view(B, S, 1+P, C)
        
        return query
