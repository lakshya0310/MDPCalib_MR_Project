"""
Pose synchronization and interpolation module.
Implements Section III-D:
- SLERP rotation interpolation
- Linear translation interpolation
- Temporal trajectory synchronization matching camera and LiDAR timestamps
"""

from typing import List, Tuple, Optional
import numpy as np
from mdpcalib.geometry.transforms import (
    normalize_quat,
    quat_to_rotmat,
    rotmat_to_quat,
    make_transform,
    invert_transform,
)


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """
    Spherical Linear Interpolation between two quaternions q0 and q1.
    alpha in [0, 1]. Format: [x, y, z, w].
    """
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)

    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    # If angle is very small, use linear interpolation
    if dot > 0.9995:
        res = q0 + alpha * (q1 - q0)
        return normalize_quat(res)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * alpha
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return normalize_quat(s0 * q0 + s1 * q1)


def interpolate_pose(
    t_target: float,
    t1: float,
    T1: np.ndarray,
    t2: float,
    T2: np.ndarray,
) -> np.ndarray:
    """
    Interpolate homogeneous pose at t_target between T1 (at t1) and T2 (at t2).
    Linear interpolation for translation, SLERP for rotation.
    """
    if abs(t2 - t1) < 1e-9 or t_target <= t1:
        return T1.copy()
    if t_target >= t2:
        return T2.copy()

    alpha = (t_target - t1) / (t2 - t1)

    # Translation linear interpolation
    pos = (1.0 - alpha) * T1[:3, 3] + alpha * T2[:3, 3]

    # Rotation SLERP
    q1 = rotmat_to_quat(T1[:3, :3])
    q2 = rotmat_to_quat(T2[:3, :3])
    q_interp = slerp(q1, q2, alpha)
    R_interp = quat_to_rotmat(q_interp)

    return make_transform(R_interp, pos)


def synchronize_trajectories(
    cam_timestamps: np.ndarray,
    cam_poses: List[np.ndarray],
    lidar_timestamps: np.ndarray,
    lidar_poses: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Synchronize LiDAR trajectory to camera timestamps.
    Returns:
    synced_cam_poses: List of 4x4 cam poses
    synced_lidar_poses: List of 4x4 interpolated LiDAR poses at camera timestamps
    valid_timestamps: Camera timestamps that fall within LiDAR timestamp range
    """
    synced_cam_poses = []
    synced_lidar_poses = []
    valid_timestamps = []

    lidar_t_min = lidar_timestamps[0]
    lidar_t_max = lidar_timestamps[-1]

    for i, t_c in enumerate(cam_timestamps):
        if t_c < lidar_t_min or t_c > lidar_t_max:
            continue

        # Find bounding LiDAR timestamps
        idx = np.searchsorted(lidar_timestamps, t_c)
        if idx == 0:
            synced_lidar = lidar_poses[0].copy()
        elif idx >= len(lidar_timestamps):
            synced_lidar = lidar_poses[-1].copy()
        else:
            t_a = lidar_timestamps[idx - 1]
            t_b = lidar_timestamps[idx]
            T_a = lidar_poses[idx - 1]
            T_b = lidar_poses[idx]
            synced_lidar = interpolate_pose(t_c, t_a, T_a, t_b, T_b)

        synced_cam_poses.append(cam_poses[i].copy())
        synced_lidar_poses.append(synced_lidar)
        valid_timestamps.append(t_c)

    return synced_cam_poses, synced_lidar_poses, np.array(valid_timestamps)


def compute_relative_motion_pairs(
    poses: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Compute relative transformation between consecutive poses:
    T_rel[i] = (P[i-1])^{-1} @ P[i]
    """
    rel_motions = []
    for i in range(1, len(poses)):
        T_prev = poses[i - 1]
        T_curr = poses[i]
        T_rel = invert_transform(T_prev) @ T_curr
        rel_motions.append(T_rel)
    return rel_motions
