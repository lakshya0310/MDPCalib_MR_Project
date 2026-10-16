"""
Motion constraint residuals for camera-LiDAR odometry matching.
Vectorized implementation of:
- Eq. (8a): Rotation error e_{i, rot} = [(R_calib * R_i^lidar)^{-1} * (R_i^cam * R_calib) - I]_{vec} in R^9
- Eq. (8b): Translation error e_{i, trans} = (R_i^cam - I) * t_calib + s_i * t_i^cam - R_calib * t_i^lidar in R^3
- Eq. (7a): J_rot with Cauchy robustifier
- Eq. (7b): J_trans with Cauchy robustifier
"""

from typing import List, Tuple, Optional, Union
import numpy as np
from mdpcalib.geometry.transforms import so3_log
from mdpcalib.optimization.robust_losses import cauchy_loss, cauchy_weight


def compute_rotation_residual(
    R_cam: np.ndarray,
    R_lidar: np.ndarray,
    R_calib: np.ndarray,
    use_matrix_form: bool = True,
) -> np.ndarray:
    """Compute single rotation residual."""
    inv_term = R_lidar.T @ R_calib.T
    rot_diff = inv_term @ (R_cam @ R_calib)

    if use_matrix_form:
        diff_mat = rot_diff - np.eye(3, dtype=np.float64)
        return diff_mat.flatten(order="F")
    else:
        return so3_log(rot_diff)


def compute_translation_residual(
    R_cam: np.ndarray,
    t_cam: np.ndarray,
    R_lidar: np.ndarray,
    t_lidar: np.ndarray,
    R_calib: np.ndarray,
    t_calib: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Compute single translation residual."""
    return (R_cam - np.eye(3)) @ t_calib.reshape(3) + scale * t_cam.reshape(3) - R_calib @ t_lidar.reshape(3)


def extract_pose_arrays(
    T_cam_list: List[np.ndarray],
    T_lidar_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-convert list of 4x4 matrices into contiguous 3D numpy arrays for vectorized math."""
    K = len(T_cam_list)
    R_cam_all = np.empty((K, 3, 3), dtype=np.float64)
    t_cam_all = np.empty((K, 3), dtype=np.float64)
    R_lid_all = np.empty((K, 3, 3), dtype=np.float64)
    t_lid_all = np.empty((K, 3), dtype=np.float64)

    for i in range(K):
        R_cam_all[i] = T_cam_list[i][:3, :3]
        t_cam_all[i] = T_cam_list[i][:3, 3]
        R_lid_all[i] = T_lidar_list[i][:3, :3]
        t_lid_all[i] = T_lidar_list[i][:3, 3]

    return R_cam_all, t_cam_all, R_lid_all, t_lid_all


def compute_motion_residuals_vec(
    R_cam_all: np.ndarray,  # (K, 3, 3)
    t_cam_all: np.ndarray,  # (K, 3)
    R_lid_all: np.ndarray,  # (K, 3, 3)
    t_lid_all: np.ndarray,  # (K, 3)
    R_calib: np.ndarray,    # (3, 3)
    t_calib: np.ndarray,    # (3,)
    scale: float,
    use_matrix_form: bool = True,
    apply_cauchy_weighting: bool = False,
    cauchy_c: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fully vectorized evaluation of motion residuals across K pairs."""
    K = R_cam_all.shape[0]
    t_c = t_calib.reshape(3)

    # 1. Rotation residual: (R_calib * R_lid)^T * (R_cam * R_calib) - I
    # R_lid^T @ R_calib^T
    term1 = np.transpose(R_lid_all, (0, 2, 1)) @ R_calib.T  # (K, 3, 3)
    # R_cam @ R_calib
    term2 = R_cam_all @ R_calib  # (K, 3, 3)
    rot_diff = term1 @ term2  # (K, 3, 3)

    if use_matrix_form:
        diff_mat = rot_diff - np.eye(3, dtype=np.float64)  # (K, 3, 3)
        # Reshape to (K, 9) with column-major order
        res_rot = np.transpose(diff_mat, (0, 2, 1)).reshape((K, 9))
    else:
        res_rot = np.empty((K, 3), dtype=np.float64)
        for i in range(K):
            res_rot[i] = so3_log(rot_diff[i])

    # 2. Translation residual: (R_cam - I) * t_calib + s * t_cam - R_calib * t_lid
    R_cam_minus_I = R_cam_all - np.eye(3, dtype=np.float64)  # (K, 3, 3)
    part1 = (R_cam_minus_I @ t_c)  # (K, 3)
    part2 = scale * t_cam_all  # (K, 3)
    part3 = (R_calib @ t_lid_all.T).T  # (K, 3)
    res_trans = part1 + part2 - part3  # (K, 3)

    if apply_cauchy_weighting:
        rot_norms = np.linalg.norm(res_rot, axis=1)
        w_r = cauchy_weight(rot_norms, c=cauchy_c)[:, np.newaxis]
        res_rot = res_rot * w_r

        trans_norms = np.linalg.norm(res_trans, axis=1)
        w_t = cauchy_weight(trans_norms, c=cauchy_c)[:, np.newaxis]
        res_trans = res_trans * w_t

    return res_rot.flatten(), res_trans.flatten()


def compute_motion_residuals(
    T_cam_list: List[np.ndarray],
    T_lidar_list: List[np.ndarray],
    R_calib: np.ndarray,
    t_calib: np.ndarray,
    scales: Union[float, List[float], np.ndarray],
    omega_rot_diag: Optional[np.ndarray] = None,
    omega_trans_diag: Optional[np.ndarray] = None,
    use_matrix_form: bool = True,
    apply_cauchy_weighting: bool = False,
    cauchy_c: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper that uses vectorized math."""
    R_cam_all, t_cam_all, R_lid_all, t_lid_all = extract_pose_arrays(T_cam_list, T_lidar_list)
    s = float(scales) if isinstance(scales, (int, float)) else float(scales[0])

    res_rot, res_trans = compute_motion_residuals_vec(
        R_cam_all, t_cam_all, R_lid_all, t_lid_all,
        R_calib, t_calib, s,
        use_matrix_form=use_matrix_form,
        apply_cauchy_weighting=apply_cauchy_weighting,
        cauchy_c=cauchy_c,
    )
    return res_rot, res_trans


def evaluate_motion_cost(
    T_cam_list: List[np.ndarray],
    T_lidar_list: List[np.ndarray],
    R_calib: np.ndarray,
    t_calib: np.ndarray,
    scales: Union[float, List[float], np.ndarray],
    cauchy_c: float = 1e-3,
    omega_rot: float = 1.0,
    omega_trans: float = 1.0,
    use_matrix_form: bool = True,
) -> Tuple[float, float, float]:
    """Vectorized cost evaluation."""
    R_cam_all, t_cam_all, R_lid_all, t_lid_all = extract_pose_arrays(T_cam_list, T_lidar_list)
    s = float(scales) if isinstance(scales, (int, float)) else float(scales[0])

    res_rot, res_trans = compute_motion_residuals_vec(
        R_cam_all, t_cam_all, R_lid_all, t_lid_all,
        R_calib, t_calib, s,
        use_matrix_form=use_matrix_form,
        apply_cauchy_weighting=False,
    )

    K = len(T_cam_list)
    rot_dim = 9 if use_matrix_form else 3
    res_rot_blocks = res_rot.reshape((K, rot_dim))
    res_trans_blocks = res_trans.reshape((K, 3))

    s_rot = np.sum(res_rot_blocks ** 2, axis=1) * omega_rot
    s_trans = np.sum(res_trans_blocks ** 2, axis=1) * omega_trans

    J_rot = float(np.sum(cauchy_loss(s_rot, c=cauchy_c)))
    J_trans = float(np.sum(cauchy_loss(s_trans, c=cauchy_c)))

    return J_rot, J_trans, J_rot + J_trans
