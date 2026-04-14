from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


inf = float("inf")


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
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

    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    quaternions = F.normalize(quaternions, p=2, dim=-1)
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def camera_to_pose_encoding(
    camera,
    pose_encoding_type="absT_quaR",
):
    """
    Inverse to pose_encoding_to_camera
    camera: opencv, cam2world
    """
    if pose_encoding_type == "absT_quaR":

        quaternion_R = matrix_to_quaternion(camera[:, :3, :3])

        pose_encoding = torch.cat([camera[:, :3, 3], quaternion_R], dim=-1)
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    return pose_encoding


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)

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


def pose_encoding_to_camera(
    pose_encoding,
    pose_encoding_type="absT_quaR",
):
    """
    Args:
        pose_encoding: A tensor of shape `BxC`, containing a batch of
                        `B` `C`-dimensional pose encodings.
        pose_encoding_type: The type of pose encoding,
    """

    if pose_encoding_type == "absT_quaR":

        abs_T = pose_encoding[:, :3]
        quaternion_R = pose_encoding[:, 3:7]
        R = quaternion_to_matrix(quaternion_R)
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    c2w_mats = torch.eye(4, 4).to(R.dtype).to(R.device)
    c2w_mats = c2w_mats[None].repeat(len(R), 1, 1)
    c2w_mats[:, :3, :3] = R
    c2w_mats[:, :3, 3] = abs_T

    return c2w_mats


def quaternion_conjugate(q):
    """Compute the conjugate of quaternion q (w, x, y, z)."""

    q_conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    return q_conj


def quaternion_multiply(q1, q2):
    """Multiply two quaternions q1 and q2."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def rotate_vector(q, v):
    """Rotate vector v by quaternion q."""
    q_vec = q[..., 1:]
    q_w = q[..., :1]

    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    v_rot = v + q_w * t + torch.cross(q_vec, t, dim=-1)
    return v_rot


def relative_pose_absT_quatR(t1, q1, t2, q2):
    """Compute the relative translation and quaternion between two poses."""

    q1_inv = quaternion_conjugate(q1)

    q_rel = quaternion_multiply(q1_inv, q2)

    delta_t = t2 - t1
    t_rel = rotate_vector(q1_inv, delta_t)
    return t_rel, q_rel
