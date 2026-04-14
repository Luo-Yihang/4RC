# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F


def extri_intri_to_pose_encoding(
    extrinsics,
    intrinsics,
    image_size_hw=None,
):
    """Convert camera extrinsics and intrinsics to a compact pose encoding."""

    # extrinsics: BxSx3x4
    # intrinsics: BxSx3x3
    R = extrinsics[:, :, :3, :3]  # BxSx3x3
    T = extrinsics[:, :, :3, 3]  # BxSx3

    quat = mat_to_quat(R)
    # Note the order of h and w here
    H, W = image_size_hw
    fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
    pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()

    return pose_encoding


def pose_encoding_to_extri_intri(
    pose_encoding,
    image_size_hw=None,
):
    """Convert a pose encoding back to camera extrinsics and intrinsics."""

    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]

    R = quat_to_mat(quat)
    extrinsics = torch.cat([R, T[..., None]], dim=-1)

    H, W = image_size_hw
    fy = (H / 2.0) / torch.clamp(torch.tan(fov_h / 2.0), 1e-6)
    fx = (W / 2.0) / torch.clamp(torch.tan(fov_w / 2.0), 1e-6)
    intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = W / 2
    intrinsics[..., 1, 2] = H / 2
    intrinsics[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1

    return extrinsics, intrinsics


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )

    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def cam_quat_xyzw_to_world_quat_wxyz(cam_quat_xyzw, c2w):
    # cam_quat_xyzw: (b, n, 4) in xyzw
    # c2w: (b, n, 4, 4)
    b, n = cam_quat_xyzw.shape[:2]
    # 1. xyzw -> wxyz
    cam_quat_wxyz = torch.cat(
        [
            cam_quat_xyzw[..., 3:4],  # w
            cam_quat_xyzw[..., 0:1],  # x
            cam_quat_xyzw[..., 1:2],  # y
            cam_quat_xyzw[..., 2:3],  # z
        ],
        dim=-1,
    )
    # 2. Quaternion to matrix
    cam_quat_wxyz_flat = cam_quat_wxyz.reshape(-1, 4)
    rotmat_cam = quat_to_mat(cam_quat_wxyz_flat).reshape(b, n, 3, 3)
    # 3. Transform to world space
    rotmat_c2w = c2w[..., :3, :3]
    rotmat_world = torch.matmul(rotmat_c2w, rotmat_cam)
    # 4. Matrix to quaternion (wxyz)
    rotmat_world_flat = rotmat_world.reshape(-1, 3, 3)
    world_quat_wxyz_flat = mat_to_quat(rotmat_world_flat)
    world_quat_wxyz = world_quat_wxyz_flat.reshape(b, n, 4)
    return world_quat_wxyz

################################################################################


from types import SimpleNamespace
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum


def as_homogeneous(ext):
    """
    Accept (..., 3,4) or (..., 4,4) extrinsics, return (...,4,4) homogeneous matrix.
    Supports torch.Tensor or np.ndarray.
    """
    if isinstance(ext, torch.Tensor):
        # If already in homogeneous form
        if ext.shape[-2:] == (4, 4):
            return ext
        elif ext.shape[-2:] == (3, 4):
            # Create a new homogeneous matrix
            ones = torch.zeros_like(ext[..., :1, :4])
            ones[..., 0, 3] = 1.0
            return torch.cat([ext, ones], dim=-2)
        else:
            raise ValueError(f"Invalid shape for torch.Tensor: {ext.shape}")

    elif isinstance(ext, np.ndarray):
        if ext.shape[-2:] == (4, 4):
            return ext
        elif ext.shape[-2:] == (3, 4):
            ones = np.zeros_like(ext[..., :1, :4])
            ones[..., 0, 3] = 1.0
            return np.concatenate([ext, ones], axis=-2)
        else:
            raise ValueError(f"Invalid shape for np.ndarray: {ext.shape}")

    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray.")


@torch.jit.script
def affine_inverse(A: torch.Tensor):
    R = A[..., :3, :3]  # ..., 3, 3
    T = A[..., :3, 3:]  # ..., 3, 1
    P = A[..., 3:, :]  # ..., 1, 4
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)


def transpose_last_two_axes(arr):
    """
    for np < 2
    """
    if arr.ndim < 2:
        return arr
    axes = list(range(arr.ndim))
    # swap the last two
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return arr.transpose(axes)


def affine_inverse_np(A: np.ndarray):
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    return np.concatenate(
        [
            np.concatenate([transpose_last_two_axes(R), -transpose_last_two_axes(R) @ T], axis=-1),
            P,
        ],
        axis=-2,
    )


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def sample_image_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> tuple[
    torch.Tensor,  # float coordinates (xy indexing), "*shape dim"
    torch.Tensor,  # integer indices (ij indexing), "*shape dim"
]:
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""

    # Each entry is a pixel-wise integer coordinate. In the 2D case, each entry is a
    # (row, col) coordinate.
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

    # Each entry is a floating-point coordinate in the range (0, 1). In the 2D case,
    # each entry is an (x, y) coordinate.
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

    return coordinates, stacked_indices


def homogenize_points(points: torch.Tensor) -> torch.Tensor:  # "*batch dim"  # "*batch dim+1"
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(vectors: torch.Tensor) -> torch.Tensor:  #  "*batch dim"  # "*batch dim+1"
    """Convert batched vectors (xyz) to (xyz0)."""
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
    homogeneous_coordinates: torch.Tensor,  # "*#batch dim"
    transformation: torch.Tensor,  # "*#batch dim dim"
) -> torch.Tensor:  # "*batch dim"
    """Apply a rigid-body transformation to points or vectors."""
    return einsum(
        transformation,
        homogeneous_coordinates.to(transformation.dtype),
        "... i j, ... j -> ... i",
    )


def transform_cam2world(
    homogeneous_coordinates: torch.Tensor,  # "*#batch dim"
    extrinsics: torch.Tensor,  # "*#batch dim dim"
) -> torch.Tensor:  # "*batch dim"
    """Transform points from 3D camera coordinates to 3D world coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics)


def unproject(
    coordinates: torch.Tensor,  # "*#batch dim"
    z: torch.Tensor,  # "*#batch"
    intrinsics: torch.Tensor,  # "*#batch dim+1 dim+1"
) -> torch.Tensor:  # "*batch dim+1"
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    coordinates = homogenize_points(coordinates)
    ray_directions = einsum(
        intrinsics.float().inverse().to(intrinsics),
        coordinates.to(intrinsics.dtype),
        "... i j, ... j -> ... i",
    )

    # Apply the supplied depth values.
    return ray_directions * z[..., None]


def get_world_rays(
    coordinates: torch.Tensor,  # "*#batch dim"
    extrinsics: torch.Tensor,  # "*#batch dim+2 dim+2"
    intrinsics: torch.Tensor,  # "*#batch dim+1 dim+1"
) -> tuple[
    torch.Tensor,  # origins, "*batch dim+1"
    torch.Tensor,  # directions, "*batch dim+1"
]:
    # Get camera-space ray directions.
    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Transform ray directions to world coordinates.
    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :-1]

    # Tile the ray origins to have the same shape as the ray directions.
    origins = extrinsics[..., :-1, -1].broadcast_to(directions.shape)

    return origins, directions


def get_fov(intrinsics: torch.Tensor) -> torch.Tensor:  # "batch 3 3" -> "batch 2"
    intrinsics_inv = intrinsics.float().inverse().to(intrinsics)

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=intrinsics.dtype, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)


def map_pdf_to_opacity(
    pdf: torch.Tensor,  # " *batch"
    global_step: int = 0,
    opacity_mapping: Optional[dict] = None,
) -> torch.Tensor:  # " *batch"
    # https://www.desmos.com/calculator/opvwti3ba9

    # Figure out the exponent.
    if opacity_mapping is not None:
        cfg = SimpleNamespace(**opacity_mapping)
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
    else:
        x = 0.0
    exponent = 2**x

    # Map the probability density to an opacity.
    return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

def normalize_homogenous_points(points):
    """Normalize the point vectors"""
    return points / points[..., -1:]

def inverse_intrinsic_matrix(ixts):
    """ """
    return torch.inverse(ixts)

def pixel_space_to_camera_space(pixel_space_points, depth, intrinsics):
    """
    Convert pixel space points to camera space points.

    Args:
        pixel_space_points (torch.Tensor): Pixel space points with shape (h, w, 2)
        depth (torch.Tensor): Depth map with shape (b, v, h, w, 1)
        intrinsics (torch.Tensor): Camera intrinsics with shape (b, v, 3, 3)

    Returns:
        torch.Tensor: Camera space points with shape (b, v, h, w, 3).
    """
    pixel_space_points = homogenize_points(pixel_space_points)
    # camera_space_points = torch.einsum(
    #     "b v i j , h w j -> b v h w i", intrinsics.inverse(), pixel_space_points
    # )
    camera_space_points = torch.einsum(
        "b v i j , h w j -> b v h w i", inverse_intrinsic_matrix(intrinsics), pixel_space_points
    )
    camera_space_points = camera_space_points * depth
    return camera_space_points


def camera_space_to_world_space(camera_space_points, c2w):
    """
    Convert camera space points to world space points.

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """
    camera_space_points = homogenize_points(camera_space_points)
    world_space_points = torch.einsum("b v i j , b v h w j -> b v h w i", c2w, camera_space_points)
    return world_space_points[..., :3]


def camera_space_to_pixel_space(camera_space_points, intrinsics):
    """
    Convert camera space points to pixel space points.

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v1, v2, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v2, 3, 3)

    Returns:
        torch.Tensor: World space points with shape (b, v1, v2, h, w, 2).
    """
    camera_space_points = normalize_homogenous_points(camera_space_points)
    pixel_space_points = torch.einsum(
        "b u i j , b v u h w j -> b v u h w i", intrinsics, camera_space_points
    )
    return pixel_space_points[..., :2]


def world_space_to_camera_space(world_space_points, c2w):
    """
    Convert world space points to pixel space points.

    Args:
        world_space_points (torch.Tensor): World space points with shape (b, v1, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v2, 4, 4)

    Returns:
        torch.Tensor: Camera space points with shape (b, v1, v2, h, w, 3).
    """
    world_space_points = homogenize_points(world_space_points)
    camera_space_points = torch.einsum(
        "b u i j , b v h w j -> b v u h w i", c2w.inverse(), world_space_points
    )
    return camera_space_points[..., :3]


def unproject_depth(
    depth, intrinsics, c2w=None, ixt_normalized=False, num_patches_x=None, num_patches_y=None
):
    """
    Turn the depth map into a 3D point cloud in world space

    Args:
        depth: (b, v, h, w, 1)
        intrinsics: (b, v, 3, 3)
        c2w: (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """
    if c2w is None:
        c2w = torch.eye(4, device=depth.device, dtype=depth.dtype)
        c2w = c2w[None, None].repeat(depth.shape[0], depth.shape[1], 1, 1)

    if not ixt_normalized:
        # Compute indices of pixels
        h, w = depth.shape[-3], depth.shape[-2]
        x_grid, y_grid = torch.meshgrid(
            torch.arange(w, device=depth.device, dtype=depth.dtype),
            torch.arange(h, device=depth.device, dtype=depth.dtype),
            indexing="xy",
        )  # (h, w), (h, w)
    else:
        # ixt_normalized: h=w=2.0. cx, cy, fx, fy are normalized according to h=w=2.0
        assert num_patches_x is not None and num_patches_y is not None
        dx = 1 / num_patches_x
        dy = 1 / num_patches_y
        max_y = 1 - dy
        min_y = -max_y
        max_x = 1 - dx
        min_x = -max_x

        grid_shift = 1.0
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(
                min_y + grid_shift,
                max_y + grid_shift,
                num_patches_y,
                dtype=torch.float32,
                device=depth.device,
            ),
            torch.linspace(
                min_x + grid_shift,
                max_x + grid_shift,
                num_patches_x,
                dtype=torch.float32,
                device=depth.device,
            ),
            indexing="ij",
        )

    # Compute coordinates of pixels in camera space
    pixel_space_points = torch.stack((x_grid, y_grid), dim=-1)  # (..., h, w, 2)
    camera_points = pixel_space_to_camera_space(
        pixel_space_points, depth, intrinsics
    )  # (..., h, w, 3)

    # Convert points to world space
    world_points = camera_space_to_world_space(camera_points, c2w)  # (..., h, w, 3)

    return world_points


def normalize_extrinsics(ex_t: torch.Tensor | None) -> torch.Tensor | None:
    """
    Normalize extrinsics to canonical coordinate system.
    
    This function:
    1. Aligns all cameras to the first camera's coordinate system
    2. Normalizes the scale based on median camera distance
    
    Args:
        ex_t: Extrinsics tensor of shape (B, N, 3, 4) or (B, N, 4, 4) or (N, 3, 4) or (N, 4, 4)
              where B is batch size, N is number of cameras
              These are world-to-camera transforms
    
    Returns:
        Normalized extrinsics with the same shape as input
    """
    if ex_t is None:
        return None
    
    # Handle both batched and unbatched inputs
    original_shape = ex_t.shape
    is_batched = ex_t.ndim == 4  # (B, N, 3/4, 4)
    
    if not is_batched:
        # Add batch dimension: (N, 3/4, 4) -> (1, N, 3/4, 4)
        ex_t = ex_t.unsqueeze(0)
    
    B, N = ex_t.shape[:2]
    
    # Ensure homogeneous form (B, N, 4, 4)
    ex_t_homog = as_homogeneous(ex_t)
    
    # Get the first camera's extrinsics for each batch: (B, 1, 4, 4)
    first_camera = ex_t_homog[:, :1, :, :]
    
    # Compute inverse to use as alignment transform
    # Reshape to (B, 4, 4) for affine_inverse
    transform = affine_inverse(first_camera.squeeze(1))  # (B, 4, 4)
    
    # Expand transform for broadcasting: (B, 1, 4, 4)
    transform = transform.unsqueeze(1)
    
    # Align all cameras to the first camera: ex_t_norm = ex_t @ transform
    # (B, N, 4, 4) @ (B, 1, 4, 4) -> (B, N, 4, 4)
    ex_t_norm = ex_t_homog @ transform
    
    # Convert to camera-to-world for distance computation
    # Reshape to (B*N, 4, 4)
    ex_t_norm_flat = ex_t_norm.reshape(B * N, 4, 4)
    c2ws_flat = affine_inverse(ex_t_norm_flat)
    c2ws = c2ws_flat.reshape(B, N, 4, 4)
    
    # Extract camera positions (translations) in world space
    translations = c2ws[..., :3, 3]  # (B, N, 3)
    
    # Compute distances from origin for each camera
    dists = translations.norm(dim=-1)  # (B, N)
    
    # Compute median distance for each batch
    median_dist = torch.median(dists, dim=-1)[0]  # (B,)
    median_dist = torch.clamp(median_dist, min=1e-1)
    
    # Normalize translation by median distance
    # Reshape for broadcasting: (B, 1, 1)
    scale_factor = median_dist.view(B, 1, 1)
    
    # Extract and normalize translation component
    # ex_t_norm is (B, N, 4, 4), we want to modify the translation part (last column, first 3 rows)
    translation = ex_t_norm[:, :, :3, 3]  # (B, N, 3)
    translation_normalized = translation / scale_factor  # (B, N, 3) / (B, 1, 1) -> (B, N, 3)
    ex_t_norm[:, :, :3, 3] = translation_normalized
    
    # Restore original shape if input was not batched
    if not is_batched:
        ex_t_norm = ex_t_norm.squeeze(0)  # (1, N, 4, 4) -> (N, 4, 4)
    
    # Restore original last two dimensions if input was (3, 4)
    if original_shape[-2] == 3:
        ex_t_norm = ex_t_norm[..., :3, :]
    
    return ex_t_norm, scale_factor


################################################################################


from einops import repeat

def compute_optimal_rotation_intrinsics_batch(
    rays_origin, rays_target, z_threshold=1e-4, reproj_threshold=0.2, weights=None,
    n_sample = None,
    n_iter=100,
    num_sample_for_ransac=8,
    rand_sample_iters_idx=None,
):
    """
    Args:
        rays_origin (torch.Tensor): (B, N, 3)
        rays_target (torch.Tensor): (B, N, 3)
        z_threshold (float): Threshold for z value to be considered valid.

    Returns:
        R (torch.tensor): (3, 3)
        focal_length (torch.tensor): (2,)
        principal_point (torch.tensor): (2,)
    """
    device = rays_origin.device
    B, N, _ = rays_origin.shape
    z_mask = torch.logical_and(
        torch.abs(rays_target[:, :, 2]) > z_threshold, torch.abs(rays_origin[:, :, 2]) > z_threshold
    ) # (B, N, 1)
    rays_origin = rays_origin.clone()
    rays_target = rays_target.clone()
    rays_origin[:, :, 0][z_mask] /= rays_origin[:, :, 2][z_mask]
    rays_origin[:, :, 1][z_mask] /= rays_origin[:, :, 2][z_mask]
    rays_target[:, :, 0][z_mask] /= rays_target[:, :, 2][z_mask]
    rays_target[:, :, 1][z_mask] /= rays_target[:, :, 2][z_mask]

    rays_origin = rays_origin[:, :, :2]
    rays_target = rays_target[:, :, :2]
    assert weights is not None, "weights must be provided"
    weights[~z_mask] = 0 

    A_list = []
    max_chunk_size = 2
    for i in range(0, rays_origin.shape[0], max_chunk_size):
        A = ransac_find_homography_weighted_fast_batch(
            rays_origin[i:i+max_chunk_size],
            rays_target[i:i+max_chunk_size],
            weights[i:i+max_chunk_size],
            n_iter=n_iter,
            n_sample = n_sample,
            num_sample_for_ransac=num_sample_for_ransac,
            reproj_threshold=reproj_threshold,
            rand_sample_iters_idx=rand_sample_iters_idx,
            max_inlier_num=8000,
        )
        A = A.to(device)
        A_need_inv_mask = torch.linalg.det(A) < 0
        A[A_need_inv_mask] = -A[A_need_inv_mask]
        A_list.append(A)

    A = torch.cat(A_list, dim=0)

    R_list = []
    f_list = []
    pp_list = []
    for i in range(A.shape[0]):
        R, L = ql_decomposition(A[i])
        L = L / L[2][2]

        f = torch.stack((L[0][0], L[1][1]))
        pp = torch.stack((L[2][0], L[2][1]))
        R_list.append(R)
        f_list.append(f)
        pp_list.append(pp)
        
    R = torch.stack(R_list)
    f = torch.stack(f_list)
    pp = torch.stack(pp_list)

    return R, f, pp


# https://www.reddit.com/r/learnmath/comments/v1crd7/linear_algebra_qr_to_ql_decomposition/
def ql_decomposition(A):
    P = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=A.device).float()
    A_tilde = torch.matmul(A, P)
    Q_tilde, R_tilde = torch.linalg.qr(A_tilde)
    Q = torch.matmul(Q_tilde, P)
    L = torch.matmul(torch.matmul(P, R_tilde), P)
    d = torch.diag(L)
    Q[:, 0] *= torch.sign(d[0])
    Q[:, 1] *= torch.sign(d[1])
    Q[:, 2] *= torch.sign(d[2])
    L[0] *= torch.sign(d[0])
    L[1] *= torch.sign(d[1])
    L[2] *= torch.sign(d[2])
    return Q, L

def find_homography_least_squares_weighted_torch(src_pts, dst_pts, confident_weight):
    """
    src_pts: (N,2) source points (torch.Tensor, float32/float64)
    dst_pts: (N,2) target points (torch.Tensor, float32/float64)
    confident_weight: (N,) weights (torch.Tensor)
    Returns: (3,3) homography matrix H (torch.Tensor)
    """
    assert src_pts.shape == dst_pts.shape
    N = src_pts.shape[0]
    if N < 4:
        raise ValueError("At least 4 points are required to compute homography.")
    assert confident_weight.shape == (N,)

    w = confident_weight.sqrt().unsqueeze(1)  # (N,1)

    x = src_pts[:, 0:1]  # (N,1)
    y = src_pts[:, 1:2]  # (N,1)
    u = dst_pts[:, 0:1]
    v = dst_pts[:, 1:2]

    zeros = torch.zeros_like(x)

    # Construct A matrix (2N, 9)
    A1 = torch.cat([-x * w, -y * w, -w, zeros, zeros, zeros, x * u * w, y * u * w, u * w], dim=1)
    A2 = torch.cat([zeros, zeros, zeros, -x * w, -y * w, -w, x * v * w, y * v * w, v * w], dim=1)
    A = torch.cat([A1, A2], dim=0)  # (2N, 9)

    # SVD
    # Note: torch.linalg.svd returns U, S, Vh, where Vh is the transpose of V
    _, _, Vh = torch.linalg.svd(A)
    H = Vh[-1].reshape(3, 3)
    H = H / H[-1, -1]
    return H


def ransac_find_homography_weighted(
    src_pts,
    dst_pts,
    confident_weight,
    n_iter=100,
    sample_ratio=0.2,
    reproj_threshold=3.0,
    num_sample_for_ransac=16,
    random_seed=None,
):
    """
    RANSAC version of weighted Homography estimation.
    Sample 4 points from the top 50% weighted points each time.
    reproj_threshold: points with reprojection error less than this value are inliers
    Returns: best_H
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    N = src_pts.shape[0]
    assert N >= 4
    # 1. Select top 50% weighted points
    sorted_idx = torch.argsort(confident_weight, descending=True)
    n_sample = max(num_sample_for_ransac, int(N * sample_ratio))
    candidate_idx = sorted_idx[:n_sample]
    best_inlier_mask = None
    best_score = 0
    for _ in range(n_iter):
        # 2. Randomly sample 4 points
        idx = candidate_idx[torch.randperm(n_sample)[:num_sample_for_ransac]]
        # 3. Compute Homography
        try:
            H = find_homography_least_squares_weighted_torch(
                src_pts[idx], dst_pts[idx], confident_weight[idx]
            )
        except Exception:
            H = torch.eye(3, dtype=src_pts.dtype, device=src_pts.device)
        # 4. Compute reprojection error for all points
        src_homo = torch.cat(
            [src_pts, torch.ones(N, 1, dtype=src_pts.dtype, device=src_pts.device)], dim=1
        )
        proj = (H @ src_homo.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        error = ((proj - dst_pts) ** 2).sum(dim=1).sqrt()  # Euclidean distance
        inlier_mask = error < reproj_threshold
        total_score = (inlier_mask * confident_weight).sum().item()
        n_inlier = inlier_mask.sum().item()
        if n_inlier < 4:
            continue  # At least 4 inliers required for fitting

        if total_score > best_score:
            best_score = total_score
            best_inlier_mask = inlier_mask

    # 5. Refit Homography using inliers
    H_inlier = find_homography_least_squares_weighted_torch(
        src_pts[best_inlier_mask], dst_pts[best_inlier_mask], confident_weight[best_inlier_mask]
    )

    return H_inlier


def find_homography_least_squares_weighted_torch_batch(
    src_pts_batch, dst_pts_batch, confident_weight_batch
):
    """
    Batch version of weighted least squares Homography
    src_pts_batch: (B, K, 2)
    dst_pts_batch: (B, K, 2)
    confident_weight_batch: (B, K)
    Returns: (B, 3, 3)
    """
    B, K, _ = src_pts_batch.shape
    w = confident_weight_batch.sqrt().unsqueeze(2)  # (B,K,1)
    x = src_pts_batch[:, :, 0:1]
    y = src_pts_batch[:, :, 1:2]
    u = dst_pts_batch[:, :, 0:1]
    v = dst_pts_batch[:, :, 1:2]
    zeros = torch.zeros_like(x)
    A1 = torch.cat([-x * w, -y * w, -w, zeros, zeros, zeros, x * u * w, y * u * w, u * w], dim=2)
    A2 = torch.cat([zeros, zeros, zeros, -x * w, -y * w, -w, x * v * w, y * v * w, v * w], dim=2)
    A = torch.cat([A1, A2], dim=1)  # (B, 2K, 9)
    # SVD: torch.linalg.svd supports batch
    _, _, Vh = torch.linalg.svd(A)
    H = Vh[:, -1].reshape(B, 3, 3)
    H = H / H[:, 2:3, 2:3]
    return H


def ransac_find_homography_weighted_fast(
    src_pts,
    dst_pts,
    confident_weight,
    n_sample,
    n_iter=100,
    reproj_threshold=3.0,
    num_sample_for_ransac=8,
    random_seed=None,
    rand_sample_iters_idx=None,
):
    """
    Batch version of RANSAC weighted Homography estimation.
    Returns: H_inlier
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    N = src_pts.shape[0]
    device = src_pts.device
    assert N >= 4
    # 1. Select top weighted points by sample_ratio
    sorted_idx = torch.argsort(confident_weight, descending=True)
    candidate_idx = sorted_idx[:n_sample]  # (n_sample,)
    if rand_sample_iters_idx is None:
        rand_sample_iters_idx = torch.stack(
            [torch.randperm(n_sample, device=device)[:num_sample_for_ransac] for _ in range(n_iter)],
            dim=0,
        )  # (n_iter, num_sample_for_ransac)
    # 2. Generate all sampling groups at once
    # shape: (n_iter, num_sample_for_ransac)
    rand_idx = candidate_idx[rand_sample_iters_idx]  # (n_iter, num_sample_for_ransac)
    # 3. Construct batch input
    src_pts_batch = src_pts[rand_idx]  # (n_iter, num_sample_for_ransac, 2)
    dst_pts_batch = dst_pts[rand_idx]  # (n_iter, num_sample_for_ransac, 2)
    confident_weight_batch = confident_weight[rand_idx]  # (n_iter, num_sample_for_ransac)
    # 4. Batch fit Homography
    H_batch = find_homography_least_squares_weighted_torch_batch(
        src_pts_batch, dst_pts_batch, confident_weight_batch
    )  # (n_iter, 3, 3)
    # 5. Batch evaluate inliers for all H
    src_homo = torch.cat(
        [src_pts, torch.ones(N, 1, dtype=src_pts.dtype, device=src_pts.device)], dim=1
    )  # (N,3)
    src_homo_expand = src_homo.unsqueeze(0).expand(n_iter, N, 3)  # (n_iter, N, 3)
    dst_pts_expand = dst_pts.unsqueeze(0).expand(n_iter, N, 2)  # (n_iter, N, 2)
    confident_weight_expand = confident_weight.unsqueeze(0).expand(n_iter, N)  # (n_iter, N)
    # H_batch: (n_iter, 3, 3)
    proj = torch.bmm(src_homo_expand, H_batch.transpose(1, 2))  # (n_iter, N, 3)
    proj_xy = proj[:, :, :2] / proj[:, :, 2:3]  # (n_iter, N, 2)
    error = ((proj_xy - dst_pts_expand) ** 2).sum(dim=2).sqrt()  # (n_iter, N)
    inlier_mask = error < reproj_threshold  # (n_iter, N)
    total_score = (inlier_mask * confident_weight_expand).sum(dim=1)  # (n_iter,)
    # 6. Select the sampling group with the highest score
    best_idx = torch.argmax(total_score)
    best_inlier_mask = inlier_mask[best_idx]  # (N,)
    inlier_src_pts = src_pts[best_inlier_mask]
    inlier_dst_pts = dst_pts[best_inlier_mask]
    inlier_confident_weight = confident_weight[best_inlier_mask]

    max_inlier_num = 10000
    sorted_idx = torch.argsort(inlier_confident_weight, descending=True)

    # method 1: sort according to confident_weight, and only keep max_inlier_num pts
    # sorted_idx = sorted_idx[:max_inlier_num]

    # method 2: random choose max_inlier_num pts
    sorted_idx = sorted_idx[torch.randperm(len(sorted_idx))[:max_inlier_num]]

    inlier_src_pts = inlier_src_pts[sorted_idx]
    inlier_dst_pts = inlier_dst_pts[sorted_idx]
    inlier_confident_weight = inlier_confident_weight[sorted_idx]
    # 7. Refit Homography using inliers
    H_inlier = find_homography_least_squares_weighted_torch(
        inlier_src_pts, inlier_dst_pts, inlier_confident_weight
    )
    return H_inlier


def ransac_find_homography_weighted_fast_batch(
    src_pts,  # (B, N, 3)
    dst_pts,  # (B, N, 2)
    confident_weight,  # (B, N)
    n_sample,
    n_iter=100,
    reproj_threshold=3.0,
    num_sample_for_ransac=8,
    max_inlier_num=10000,
    random_seed=None,
    rand_sample_iters_idx=None,
):
    """
    Batch version of RANSAC weighted Homography estimation (supports batch).
    Input:
        src_pts: (B, N, 2)
        dst_pts: (B, N, 2)
        confident_weight: (B, N)
    Returns:
        H_inlier: (B, 3, 3)
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    B, N, _ = src_pts.shape
    assert N >= 4

    device = src_pts.device

    # 1. Select top weighted points by sample_ratio
    sorted_idx = torch.argsort(confident_weight, descending=True, dim=1)  # (B, N)
    candidate_idx = sorted_idx[:, :n_sample]  # (B, n_sample)

    # 2. Generate all sampling groups at once
    # rand_idx: (B, n_iter, num_sample_for_ransac)
    if rand_sample_iters_idx is None:
        rand_sample_iters_idx = torch.stack(
            [torch.randperm(n_sample, device=device)[:num_sample_for_ransac] for _ in range(n_iter)],
            dim=0,
        )  # (n_iter, num_sample_for_ransac)
    
    rand_idx = candidate_idx[:, rand_sample_iters_idx]  # (B, n_iter, num_sample_for_ransac)

    # 3. Construct batch input
    # Indexing method below: (B, n_iter, num_sample_for_ransac, ...)
    b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, n_iter, num_sample_for_ransac)
    src_pts_batch = src_pts[b_idx, rand_idx]  # (B, n_iter, num_sample_for_ransac, 2)
    dst_pts_batch = dst_pts[b_idx, rand_idx]  # (B, n_iter, num_sample_for_ransac, 2)
    confident_weight_batch = confident_weight[b_idx, rand_idx]  # (B, n_iter, num_sample_for_ransac)

    # 4. Batch fit Homography
    # Need to implement batch version that supports (B, n_iter, num_sample_for_ransac, ...) input
    # Output H_batch: (B, n_iter, 3, 3)
    cB, cN = src_pts_batch.shape[:2]
    H_batch = find_homography_least_squares_weighted_torch_batch(
        src_pts_batch.flatten(0, 1), dst_pts_batch.flatten(0, 1), confident_weight_batch.flatten(0, 1)
    )  # (B, n_iter, 3, 3)
    H_batch = H_batch.unflatten(0, (cB, cN))

    # 5. Batch evaluate inliers for all H
    src_homo = torch.cat(
        [src_pts, torch.ones(B, N, 1, dtype=src_pts.dtype, device=src_pts.device)], dim=2
    )  # (B, N, 3)
    src_homo_expand = src_homo.unsqueeze(1).expand(B, n_iter, N, 3)  # (B, n_iter, N, 3)
    dst_pts_expand = dst_pts.unsqueeze(1).expand(B, n_iter, N, 2)  # (B, n_iter, N, 2)
    confident_weight_expand = confident_weight.unsqueeze(1).expand(B, n_iter, N)  # (B, n_iter, N)

    # H_batch: (B, n_iter, 3, 3)
    # Need to reshape H_batch to (B*n_iter, 3, 3), src_homo_expand to (B*n_iter, N, 3)
    H_batch_flat = H_batch.reshape(-1, 3, 3)
    src_homo_expand_flat = src_homo_expand.reshape(-1, N, 3)
    proj = torch.bmm(src_homo_expand_flat, H_batch_flat.transpose(1, 2))  # (B*n_iter, N, 3)
    proj_xy = proj[:, :, :2] / proj[:, :, 2:3]  # (B*n_iter, N, 2)
    proj_xy = proj_xy.reshape(B, n_iter, N, 2)
    error = ((proj_xy - dst_pts_expand) ** 2).sum(dim=3).sqrt()  # (B, n_iter, N)
    inlier_mask = error < reproj_threshold  # (B, n_iter, N)
    total_score = (inlier_mask * confident_weight_expand).sum(dim=2)  # (B, n_iter)

    # 6. Select the sampling group with the highest score
    best_idx = torch.argmax(total_score, dim=1)  # (B,)
    best_inlier_mask = inlier_mask[torch.arange(B, device=device), best_idx]  # (B, N)

    # 7. Refit Homography using inliers
    H_inlier_list = []
    for b in range(B):
        mask = best_inlier_mask[b]
        inlier_src_pts = src_pts[b][mask]  # (?, 3)
        inlier_dst_pts = dst_pts[b][mask]  # (?, 2)
        inlier_confident_weight = confident_weight[b][mask]  # (?)

        sorted_idx = torch.argsort(inlier_confident_weight, descending=True)
        # # method 1: sort according to confident_weight, and only keep max_inlier_num pts
        # sorted_idx = sorted_idx[:max_inlier_num]
        # method 2: random choose max_inlier_num pts
        if len(sorted_idx) > max_inlier_num:
            # random choose from first 95% confident pts
            keep_len = max(int(len(sorted_idx) * 0.95), max_inlier_num)
            sorted_idx = sorted_idx[:keep_len]
            perm = torch.randperm(len(sorted_idx), device=device)[:max_inlier_num]
            sorted_idx = sorted_idx[perm]
        inlier_src_pts = inlier_src_pts[sorted_idx]
        inlier_dst_pts = inlier_dst_pts[sorted_idx]
        inlier_confident_weight = inlier_confident_weight[sorted_idx]

        H_inlier = find_homography_least_squares_weighted_torch(
            inlier_src_pts, inlier_dst_pts, inlier_confident_weight
        )  # (3, 3)
        H_inlier_list.append(H_inlier)
    H_inlier = torch.stack(H_inlier_list, dim=0)  # (B, 3, 3)
    return H_inlier

def get_params_for_ransac(N, device):
    n_iter=100
    sample_ratio=0.3
    num_sample_for_ransac=8
    n_sample = max(num_sample_for_ransac, int(N * sample_ratio))
    rand_sample_iters_idx = torch.stack(
            [torch.randperm(n_sample, device=device)[:num_sample_for_ransac] for _ in range(n_iter)],
            dim=0,
        )  # (n_iter, num_sample_for_ransac)
    return n_iter, num_sample_for_ransac, n_sample, rand_sample_iters_idx


def camray_to_caminfo(camray, confidence=None, reproj_threshold=0.2, training=False):
    """
    Args:
        camray: (B, S, num_patches_y, num_patches_x, 6)
        confidence: (B, S, num_patches_y, num_patches_x)
    Returns:
        R: (B, S, 3, 3)
        T: (B, S, 3)
        focal_lengths: (B, S, 2)
        principal_points: (B, S, 2)
    """
    if confidence is None:
        confidence = torch.ones_like(camray[:, :, :, :, 0])
    B, S, num_patches_y, num_patches_x, _ = camray.shape
    # identity K, assume imw=imh=2.0
    I_K = torch.eye(3, dtype=camray.dtype, device=camray.device)
    I_K[0, 2] = 1.0
    I_K[1, 2] = 1.0
    # repeat I_K to match camray
    I_K = I_K.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)

    cam_plane_depth = torch.ones(
        B, S, num_patches_y, num_patches_x, 1, dtype=camray.dtype, device=camray.device
    )
    I_cam_plane_unproj = unproject_depth(
        cam_plane_depth,
        I_K,
        c2w=None,
        ixt_normalized=True,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
    )  # (B, S, num_patches_y, num_patches_x, 3)

    camray = camray.flatten(0, 1).flatten(1, 2)  # (B*S, num_patches_y*num_patches_x, 6)
    I_cam_plane_unproj = I_cam_plane_unproj.flatten(0, 1).flatten(
        1, 2
    )  # (B*S, num_patches_y*num_patches_x, 3)
    confidence = confidence.flatten(0, 1).flatten(1, 2)  # (B*S, num_patches_y*num_patches_x)
    
    # Compute optimal rotation to align rays
    N = camray.shape[-2]
    device = camray.device
    n_iter, num_sample_for_ransac, n_sample, rand_sample_iters_idx = get_params_for_ransac(N, device)
    
    # Use batch processing (confidence is guaranteed to be not None at this point)
    if training:
        camray = camray.clone().detach()
        I_cam_plane_unproj = I_cam_plane_unproj.clone().detach()
        confidence = confidence.clone().detach()
    R, focal_lengths, principal_points = compute_optimal_rotation_intrinsics_batch(
        I_cam_plane_unproj,
        camray[:, :, :3],
        reproj_threshold=reproj_threshold,
        weights=confidence,
        n_sample = n_sample,
        n_iter=n_iter,
        num_sample_for_ransac=num_sample_for_ransac,
        rand_sample_iters_idx=rand_sample_iters_idx,
    )

    T = torch.sum(camray[:, :, 3:] * confidence.unsqueeze(-1), dim=1) / torch.sum(
        confidence, dim=-1, keepdim=True
    )

    R = R.reshape(B, S, 3, 3)
    T = T.reshape(B, S, 3)
    focal_lengths = focal_lengths.reshape(B, S, 2)
    principal_points = principal_points.reshape(B, S, 2)

    return R, T, 1.0 / focal_lengths, principal_points + 1.0

def get_extrinsic_from_camray(camray, conf, patch_size_y, patch_size_x, training=False):
    pred_R, pred_T, pred_focal_lengths, pred_principal_points = camray_to_caminfo(
        camray, confidence=conf.squeeze(-1), training=training
    )

    pred_extrinsic = torch.cat(
        [
            torch.cat([pred_R, pred_T.unsqueeze(-1)], dim=-1),
            repeat(
                torch.tensor([0, 0, 0, 1], dtype=pred_R.dtype, device=pred_R.device),
                "c -> b s 1 c",
                b=pred_R.shape[0],
                s=pred_R.shape[1],
            ),
        ],
        dim=-2,
    )  # B, S, 4, 4
    return pred_extrinsic, pred_focal_lengths, pred_principal_points