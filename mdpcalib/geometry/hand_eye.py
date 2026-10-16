"""
Closed-form hand-eye calibration solvers:
Solves AX = XB (or A_i X = X B_i) with scale factor s:
R_A * R_X = R_X * R_B
R_A * t_X + s * t_A = R_X * t_B + t_X

Implements:
- Shiu & Ahmad [31] / Tsai-Lenz rotation vector alignment via SVD
- Park & Martin Lie-algebraic solution
- Linear least squares for translation and monocular scale factor s
"""

from typing import List, Tuple, Optional
import numpy as np
from mdpcalib.geometry.transforms import (
    so3_log,
    rotmat_to_quat,
    quat_to_rotmat,
    make_transform,
)


def solve_rotation_svd(
    R_cam_list: List[np.ndarray],
    R_lidar_list: List[np.ndarray],
) -> np.ndarray:
    """
    Solve R_cam * R_X = R_X * R_lidar using rotation axis matching and SVD (Kabsch / Shiu-Ahmad).
    For each pair, rotation axis vectors satisfy: axis_cam = R_X * axis_lidar.
    """
    K = len(R_cam_list)
    H = np.zeros((3, 3), dtype=np.float64)

    for i in range(K):
        w_cam = so3_log(R_cam_list[i])
        w_lid = so3_log(R_lidar_list[i])
        theta_cam = np.linalg.norm(w_cam)
        theta_lid = np.linalg.norm(w_lid)

        # Skip nearly zero rotations (degenerate axes)
        if theta_cam < 1e-4 or theta_lid < 1e-4:
            continue

        u_cam = w_cam / theta_cam
        u_lid = w_lid / theta_lid

        # Weight by rotation magnitude
        w = 0.5 * (theta_cam + theta_lid)
        H += w * np.outer(u_cam, u_lid)

    U, S, Vt = np.linalg.svd(H)
    R_X = U @ Vt
    # Enforce det(R) = +1 for valid SO(3)
    if np.linalg.det(R_X) < 0:
        Vt[-1, :] *= -1
        R_X = U @ Vt

    return R_X


def solve_translation_and_scale(
    R_cam_list: List[np.ndarray],
    t_cam_list: List[np.ndarray],
    R_lidar_list: List[np.ndarray],
    t_lidar_list: List[np.ndarray],
    R_X: np.ndarray,
    estimate_scale: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Solve for translation t_X and scale s given R_X:
    (R_cam_i - I) * t_X + s * t_cam_i = R_X * t_lidar_i
    
    If estimate_scale=True, solves for [t_X (3x1); s (1x1)] in R^4.
    If estimate_scale=False, assumes s=1.0 and solves for t_X in R^3.
    """
    K = len(R_cam_list)
    A_blocks = []
    b_blocks = []

    for i in range(K):
        R_c = R_cam_list[i]
        t_c = t_cam_list[i].reshape(3, 1)
        t_l = t_lidar_list[i].reshape(3, 1)

        # LHS matrix
        # (R_c - I) * t_X + s * t_c = R_X * t_l
        A_rot = R_c - np.eye(3)
        if estimate_scale:
            A_row = np.hstack([A_rot, t_c])  # (3, 4)
        else:
            A_row = A_rot  # (3, 3)

        b_row = R_X @ t_l
        if not estimate_scale:
            b_row -= t_c

        A_blocks.append(A_row)
        b_blocks.append(b_row)

    if not A_blocks:
        return np.zeros(3, dtype=np.float64), 1.0

    A = np.vstack(A_blocks)
    b = np.vstack(b_blocks)

    # Solve via SVD / least squares
    sol, residuals, rank, s_vals = np.linalg.lstsq(A, b, rcond=None)
    sol = sol.flatten()

    if estimate_scale:
        t_X = sol[:3]
        scale = float(sol[3])
        if scale <= 0:
            scale = 1.0
    else:
        t_X = sol[:3]
        scale = 1.0

    return t_X, scale


def solve_hand_eye_shiu_ahmad(
    T_cam_list: List[np.ndarray],
    T_lidar_list: List[np.ndarray],
    estimate_scale: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Complete closed-form hand-eye calibration (Shiu-Ahmad / Tsai-Lenz formulation).
    Returns:
    T_calib: 4x4 transform
    scale: estimated scale factor s
    """
    R_cam = [T[:3, :3] for T in T_cam_list]
    t_cam = [T[:3, 3] for T in T_cam_list]
    R_lid = [T[:3, :3] for T in T_lidar_list]
    t_lid = [T[:3, 3] for T in T_lidar_list]

    R_X = solve_rotation_svd(R_cam, R_lid)
    t_X, s = solve_translation_and_scale(R_cam, t_cam, R_lid, t_lid, R_X, estimate_scale=estimate_scale)

    return make_transform(R_X, t_X), s


def solve_hand_eye_park_martin(
    T_cam_list: List[np.ndarray],
    T_lidar_list: List[np.ndarray],
    estimate_scale: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Park and Martin hand-eye calibration formulation using Lie algebra.
    """
    K = len(T_cam_list)
    A_mat = np.zeros((3, 3), dtype=np.float64)

    for i in range(K):
        alpha = so3_log(T_cam_list[i][:3, :3])
        beta = so3_log(T_lidar_list[i][:3, :3])
        A_mat += np.outer(beta, alpha)

    # R_X = (A^T A)^{-1/2} A^T
    M = A_mat.T @ A_mat
    evals, evecs = np.linalg.eigh(M)
    evals = np.maximum(evals, 1e-12)
    inv_sqrt_M = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    R_X = inv_sqrt_M @ A_mat.T

    if np.linalg.det(R_X) < 0:
        R_X = -R_X

    R_cam = [T[:3, :3] for T in T_cam_list]
    t_cam = [T[:3, 3] for T in T_cam_list]
    R_lid = [T[:3, :3] for T in T_lidar_list]
    t_lid = [T[:3, 3] for T in T_lidar_list]

    t_X, s = solve_translation_and_scale(R_cam, t_cam, R_lid, t_lid, R_X, estimate_scale=estimate_scale)
    return make_transform(R_X, t_X), s
