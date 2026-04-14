# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# post process function for all heads: extract 3D points/confidence from output
# --------------------------------------------------------
import torch


def reg_desc(desc, mode):
    if 'norm' in mode:
        desc = desc / desc.norm(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown desc mode {mode}")
    return desc


def postprocess_with_feature(out, depth_mode, conf_mode, desc_dim=None, desc_mode='norm', two_confs=False, desc_conf_mode=None):
    if desc_conf_mode is None:
        desc_conf_mode = conf_mode
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,D
    res = dict(pts3d=reg_dense_depth(fmap[..., 0:3], mode=depth_mode))
    if conf_mode is not None:
        res['conf'] = reg_dense_conf(fmap[..., 3], mode=conf_mode)
    if desc_dim is not None:
        start = 3 + int(conf_mode is not None)
        res['desc'] = reg_desc(fmap[..., start:start + desc_dim], mode=desc_mode)
        if two_confs:
            res['desc_conf'] = reg_dense_conf(fmap[..., start + desc_dim], mode=desc_conf_mode)
        else:
            res['desc_conf'] = res['conf'].clone()
    return res


def postprocess(out, depth_mode, conf_mode, vis_mode):
    """
    extract 3D points/confidence from prediction head output
    """
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,3
    res = dict(pts3d=reg_dense_depth(fmap[:, :, :, 0:3], mode=depth_mode))

    if conf_mode is not None:
        res["conf"] = reg_dense_conf(fmap[:, :, :, 3], mode=conf_mode)
    
    if vis_mode is not None:
        res["vis"] = reg_dense_conf(fmap[:, :, :, 3:4], mode=vis_mode)

    return res


def reg_dense_depth(xyz, mode):
    """
    extract 3D points from prediction head output
    """
    mode, vmin, vmax = mode

    no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))
    assert no_bounds

    if mode == "linear":
        if no_bounds:
            return xyz  # [-inf, +inf]
        return xyz.clip(min=vmin, max=vmax)

    # distance to origin
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)

    if mode == "square":
        return xyz * d.square()

    if mode == "exp":
        return xyz * torch.expm1(d)

    raise ValueError(f"bad {mode=}")


def reg_dense_conf(x, mode):
    """
    extract confidence from prediction head output
    """
    mode, vmin, vmax = mode
    if mode == "exp":
        return vmin + x.exp().clip(max=vmax - vmin)
    if mode == "sigmoid":
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    if mode == "none":
        return x
    raise ValueError(f"bad {mode=}")
