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
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

XFORMERS_AVAILABLE = False


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning"""
    def __init__(self, in_dim: int, out_dim: int, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_B(self.lora_A(x)) * self.scaling


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
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        self.use_lora = use_lora

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        if self.use_lora:
            self.qkv_lora = LoRALayer(dim, dim * 3, lora_rank, lora_alpha)
            self.proj_lora = LoRALayer(dim, dim, lora_rank, lora_alpha)

    def forward(self, x: Tensor, pos=None, causal_mask=None, causal_view_info=None) -> Tensor:
        B, N, C = x.shape

        if self.use_lora:
            qkv = self.qkv(x) + self.qkv_lora(x)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            assert causal_mask is None or causal_view_info is None, "causal_mask and causal_view_info cannot both be non-None"
            if causal_view_info is None:
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    attn_mask=causal_mask,
                )
            else:
                V = causal_view_info
                L = N // V
                assert V*L == N

                x = torch.zeros_like(q)
                for v_q in range(V):
                    q_start = v_q * L
                    q_end = (v_q + 1) * L
                    q_view = q[:, :, q_start:q_end, :]

                    k_end = (v_q + 1) * L

                    # first 2 see
                    # if v_q == 0:
                    #     k_end = (v_q + 2) * L

                    k_view = k[:, :, :k_end, :]
                    v_view = v[:, :, :k_end, :]

                    # window
                    # window_size = 5 * L
                    # k_view = torch.cat([k[:, :, 0:L, :], k[:, :, max(k_end-window_size, L):k_end, :]], dim=2)
                    # v_view = torch.cat([v[:, :, 0:L, :], v[:, :, max(k_end-window_size, L):k_end, :]], dim=2)

                    x[:, :, q_start:q_end, :] = F.scaled_dot_product_attention(q_view, k_view, v_view)
        else:
            assert causal_mask is None, "attn_mask is not implemented for non-fused attention"
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)

        if self.use_lora:
            x = self.proj(x) + self.proj_lora(x)
        else:
            x = self.proj(x)

        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
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
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()

        assert not use_lora, "CrossAttention does not support LoRA yet"

        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.toq = nn.Linear(dim, dim, bias=qkv_bias)
        self.tok = nn.Linear(dim, dim, bias=qkv_bias)
        self.tov = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, y: Tensor, pos=None, return_attn_weights=False) -> Tensor:
        B, Nx, C = x.shape
        _, Ny, _ = y.shape

        q = self.toq(x).reshape(B, Nx, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.tok(y).reshape(B, Ny, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.tov(y).reshape(B, Ny, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            if isinstance(pos, tuple):
                pos_q, pos_k = pos
                q = self.rope(q, pos_q)
                k = self.rope(k, pos_k)
            else:
                q = self.rope(q, pos)
                k = self.rope(k, pos)

        if self.fused_attn and not return_attn_weights:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, Nx, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn_weights:
            # we hardcode start_token as 5 token here
            attn_weights = attn[:, :, 5:, 5:].sum(dim=1)
            return x, attn_weights
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
