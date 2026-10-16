"""
Point correspondence residuals for 2D pixel to 3D point matching.
Implements:
- Eq. (11): e_{j, corr} = proj(p_j^lidar, K, T_calib) - p_j^cmr in R^2
- Eq. (12): J_corr = sum_j rho_i(e_{j, corr}^T * Omega_corr * e_{j, corr})
- Fast analytical Jacobians for Gauss-Newton / Levenberg-Marquardt optimization
"""

from typing import Tuple, Optional
import numpy as np
from mdpcalib.geometry.transforms import skew_symmetric, make_transform
from mdpcalib.optimization.robust_losses import cauchy_loss, cauchy_weight


def compute_point_residual(
    pt_3d_lidar: np.ndarray,
    pt_2d_cmr: np.ndarray,
    K: np.ndarray,
    T_calib: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """
    Compute 2D reprojection error vector e_{corr} in R^2:
    e_{corr} = proj(p_lidar, K, T_calib) - p_cmr
    Returns:
    error: (2,) array
    is_valid: bool (True if point is in front of camera Z > 0.1)
    """
    p_l = pt_3d_lidar.reshape(3)
    p_cam = T_calib[:3, :3] @ p_l + T_calib[:3, 3]

    zc = p_cam[2]
    if zc < 0.1:
        return np.zeros(2, dtype=np.float64), False

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * (p_cam[0] / zc) + cx
    v = fy * (p_cam[1] / zc) + cy

    p_meas = pt_2d_cmr.reshape(2)
    err = np.array([u - p_meas[0], v - p_meas[1]], dtype=np.float64)
    return err, True


def compute_point_residuals(
    pts_3d_lidar: np.ndarray,
    pts_2d_cmr: np.ndarray,
    K: np.ndarray,
    T_calib: np.ndarray,
    omega_corr_diag: Optional[np.ndarray] = None,
    apply_cauchy_weighting: bool = False,
    cauchy_c: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute point correspondence residuals across a batch of correspondences.
    pts_3d_lidar: (N, 3)
    pts_2d_cmr: (N, 2)
    Returns:
    residuals: (M*2,) array of valid weighted residuals
    valid_mask: (N,) boolean mask
    """
    N = pts_3d_lidar.shape[0]
    if N == 0:
        return np.empty(0, dtype=np.float64), np.zeros(0, dtype=bool)

    # Transform all points to camera frame: p_cam = R * p_l + t
    R = T_calib[:3, :3]
    t = T_calib[:3, 3]
    pts_cam = (R @ pts_3d_lidar.T).T + t  # (N, 3)

    zc = pts_cam[:, 2]
    valid_mask = zc > 0.1

    if not np.any(valid_mask):
        return np.empty(0, dtype=np.float64), valid_mask

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    xc_val = pts_cam[valid_mask, 0]
    yc_val = pts_cam[valid_mask, 1]
    zc_val = pts_cam[valid_mask, 2]

    u_proj = fx * (xc_val / zc_val) + cx
    v_proj = fy * (yc_val / zc_val) + cy

    u_cmr = pts_2d_cmr[valid_mask, 0]
    v_cmr = pts_2d_cmr[valid_mask, 1]

    e_u = u_proj - u_cmr
    e_v = v_proj - v_cmr

    if omega_corr_diag is not None:
        e_u *= np.sqrt(omega_corr_diag[0])
        e_v *= np.sqrt(omega_corr_diag[1])

    errors_2d = np.stack([e_u, e_v], axis=1)  # (M, 2)

    if apply_cauchy_weighting:
        norms = np.linalg.norm(errors_2d, axis=1)  # (M,)
        weights = cauchy_weight(norms, c=cauchy_c)  # (M,)
        errors_2d = errors_2d * weights[:, np.newaxis]

    return errors_2d.flatten(), valid_mask


def evaluate_point_cost(
    pts_3d_lidar: np.ndarray,
    pts_2d_cmr: np.ndarray,
    K: np.ndarray,
    T_calib: np.ndarray,
    cauchy_c: float = 1e-3,
    omega_corr: float = 1.0,
) -> float:
    """
    Evaluate J_corr using Cauchy loss (Eq. 12).
    """
    res, _ = compute_point_residuals(
        pts_3d_lidar,
        pts_2d_cmr,
        K,
        T_calib,
        omega_corr_diag=np.array([omega_corr, omega_corr], dtype=np.float64),
        apply_cauchy_weighting=False,
    )
    if len(res) == 0:
        return 0.0

    res_2d = res.reshape((-1, 2))
    s_corr = np.sum(res_2d ** 2, axis=1)
    return float(np.sum(cauchy_loss(s_corr, c=cauchy_c)))


def compute_point_jacobian_se3(
    pts_3d_lidar: np.ndarray,
    K: np.ndarray,
    T_calib: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytical Jacobian of 2D reprojection with respect to camera-LiDAR perturbation [delta_t, omega] in R^6:
    T_perturbed = exp([delta_t, omega]) @ T_calib.
    Returns:
    J: (2*M, 6) Jacobian matrix
    valid_mask: (N,) boolean mask
    """
    N = pts_3d_lidar.shape[0]
    R = T_calib[:3, :3]
    t = T_calib[:3, 3]
    pts_cam = (R @ pts_3d_lidar.T).T + t  # (N, 3)

    zc = pts_cam[:, 2]
    valid_mask = zc > 0.1
    M = np.sum(valid_mask)
    if M == 0:
        return np.empty((0, 6), dtype=np.float64), valid_mask

    fx, fy = K[0, 0], K[1, 1]
    X = pts_cam[valid_mask, 0]
    Y = pts_cam[valid_mask, 1]
    Z = pts_cam[valid_mask, 2]

    inv_Z = 1.0 / Z
    inv_Z2 = inv_Z * inv_Z

    J = np.zeros((2 * M, 6), dtype=np.float64)

    # row 2*k: u residual w.r.t [tx, ty, tz, wx, wy, wz]
    # du/dp_cam = [fx/Z, 0, -fx*X/Z^2]
    # dp_cam / dt = I_3
    # dp_cam / d_omega = - [p_cam]_x
    J[0::2, 0] = fx * inv_Z
    J[0::2, 1] = 0.0
    J[0::2, 2] = -fx * X * inv_Z2
    J[0::2, 3] = -fx * X * Y * inv_Z2
    J[0::2, 4] = fx * (1.0 + (X * X) * inv_Z2)
    J[0::2, 5] = -fx * Y * inv_Z

    # row 2*k + 1: v residual w.r.t [tx, ty, tz, wx, wy, wz]
    # dv/dp_cam = [0, fy/Z, -fy*Y/Z^2]
    J[1::2, 0] = 0.0
    J[1::2, 1] = fy * inv_Z
    J[1::2, 2] = -fy * Y * inv_Z2
    J[1::2, 3] = -fy * (1.0 + (Y * Y) * inv_Z2)
    J[1::2, 4] = fy * X * Y * inv_Z2
    J[1::2, 5] = fy * X * inv_Z

    return J, valid_mask
