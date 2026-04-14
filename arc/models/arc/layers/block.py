# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


XFORMERS_AVAILABLE = False


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, pos=None, causal_mask=None, causal_view_info=None) -> Tensor:
        def attn_residual_func(x: Tensor, pos=None, causal_mask=None, causal_view_info=None) -> Tensor:
            if causal_mask is not None or causal_view_info is not None:
                return self.ls1(self.attn(self.norm1(x), pos=pos, causal_mask=causal_mask, causal_view_info=causal_view_info))
            else:
                return self.ls1(self.attn(self.norm1(x), pos=pos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            assert causal_mask is None and causal_view_info is None, "causal_mask or causal_view_info is not None not supported for drop_add_residual_stochastic_depth"
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                pos=pos,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos, causal_mask=causal_mask, causal_view_info=causal_view_info))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, pos=pos, causal_mask=causal_mask, causal_view_info=causal_view_info)
            x = x + ffn_residual_func(x)
        return x


class AdaLNBlock(Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            proj_bias,
            ffn_bias,
            drop,
            attn_drop,
            init_values,
            drop_path,
            act_layer,
            norm_layer,
            attn_class,
            ffn_layer,
            qk_norm,
            fused_attn,
            rope,
            use_lora,
            lora_rank,
            lora_alpha,
        )
        # remove layerscale if it exists (it shouldn't because init_values=None, but to be safe)
        if hasattr(self, "ls1"):
            del self.ls1
        if hasattr(self, "ls2"):
            del self.ls2
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        # zero init
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: Tensor, cond: Tensor, pos=None, attn_mask=None) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)

        def attn_residual_func(x: Tensor, pos=None) -> Tensor:
            x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
            if attn_mask is not None:
                return gate_msa.unsqueeze(1) * self.attn(x_norm, pos=pos, attn_mask=attn_mask)
            else:
                return gate_msa.unsqueeze(1) * self.attn(x_norm, pos=pos)

        def ffn_residual_func(x: Tensor) -> Tensor:
            x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
            return gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        # Drop path logic
        # Note: drop_path implementation in Block assumes attn_residual_func signature is (x, pos).
        # But here modulation params are closure-captured. That should be fine.
        
        if self.training and self.sample_drop_ratio > 0.1:
            x = drop_add_residual_stochastic_depth(
                x,
                pos=pos,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            x = x + self.drop_path1(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x, pos=pos)
            x = x + ffn_residual_func(x)
        return x


class CrossBlock(Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        cross_attn_init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        cross_has_residual: bool = True,
        cross_detach_kv: bool = False,
    ) -> None:
        super().__init__(
            dim=dim, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            proj_bias=proj_bias, 
            ffn_bias=ffn_bias, 
            drop=drop, 
            attn_drop=attn_drop, 
            init_values=init_values, 
            drop_path=drop_path, 
            act_layer=act_layer, 
            norm_layer=norm_layer, 
            attn_class=attn_class, 
            ffn_layer=ffn_layer, 
            qk_norm=qk_norm, 
            fused_attn=fused_attn, 
            rope=rope,
        )
        self.normy = norm_layer(dim)
        self.cross_has_residual = cross_has_residual
        self.cross_detach_kv = cross_detach_kv

        self.ls1 = LayerScale(dim, init_values=cross_attn_init_values) if cross_attn_init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x: Tensor, y: Tensor, pos=None, return_attn_weights=False) -> Tensor:
        def attn_residual_func(x: Tensor, y: Tensor, pos=None) -> Tensor:
            if return_attn_weights:
                x, attn_weights = self.attn(self.norm1(x), self.normy(y), pos=pos, return_attn_weights=return_attn_weights)
                return self.ls1(x), attn_weights
            else:
                return self.ls1(self.attn(self.norm1(x), self.normy(y), pos=pos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.cross_detach_kv:
            y = y.detach()

        if self.training and self.sample_drop_ratio > 0.1:
            assert self.sample_drop_ratio == 0
            # this seems will not run as sample_drop_ratio == 0
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                pos=pos,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            assert self.sample_drop_ratio == 0
            if self.cross_has_residual:
                x = x + self.drop_path1(attn_residual_func(x, y, pos=pos))
            else:
                x = attn_residual_func(x, y, pos=pos)
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            if self.cross_has_residual:
                if return_attn_weights:
                    delta, attn_weights = attn_residual_func(x, y, pos=pos)
                    x = x + delta
                else:
                    x = x + attn_residual_func(x, y, pos=pos)
            else:
                # we only implement return_attn_weights in the else for now
                if return_attn_weights:
                    x, attn_weights = attn_residual_func(x, y, pos=pos)
                else:
                    x = attn_residual_func(x, y, pos=pos)
            x = x + ffn_residual_func(x)

        if return_attn_weights:
            return x, attn_weights
        else:
            return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
    pos=None,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError
