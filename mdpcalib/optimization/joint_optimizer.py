"""
Joint Registration Optimizer for MDPCalib.
Solves Eq. (13):
T_final^calib = argmin_{T_calib} (J_mot + J_corr)
jointly minimizing sensor motion costs and 2D-3D point correspondence costs with Cauchy robust loss.
"""

from typing import List, Tuple, Optional, Union
import numpy as np
from mdpcalib.geometry.transforms import (
    so3_exp,
    so3_log,
    make_transform,
    invert_transform,
)
from mdpcalib.optimization.robust_losses import cauchy_loss, cauchy_weight
from mdpcalib.optimization.motion_residuals import (
    compute_motion_residuals,
    evaluate_motion_cost,
)
from mdpcalib.optimization.point_residuals import (
    compute_point_residuals,
    evaluate_point_cost,
    compute_point_jacobian_se3,
)


class JointOptimizer:
    """
    Jointly optimizes extrinsic camera-LiDAR transformation T_calib w.r.t.
    both sensor motion constraints (J_mot) and deep point correspondences (J_corr).
    """

    def __init__(
        self,
        cauchy_param: float = 1e-3,
        omega_rot: float = 1.0,
        omega_trans: float = 1.0,
        omega_corr: float = 1.0,
        max_iterations: int = 150,
        tolerance: float = 1e-8,
        use_matrix_form: bool = True,
        include_motion: bool = True,
        use_cauchy: bool = True,
    ):
        self.cauchy_param = cauchy_param
        self.omega_rot = omega_rot
        self.omega_trans = omega_trans
        self.omega_corr = omega_corr
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.use_matrix_form = use_matrix_form
        self.include_motion = include_motion
        self.use_cauchy = use_cauchy

    def evaluate_total_cost(
        self,
        R_calib: np.ndarray,
        t_calib: np.ndarray,
        scale: float,
        T_cam_list: List[np.ndarray],
        T_lidar_list: List[np.ndarray],
        pts_3d: np.ndarray,
        pts_2d: np.ndarray,
        K: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Evaluate J_mot, J_corr, and total J_total = J_mot + J_corr.
        """
        T_curr = make_transform(R_calib, t_calib)

        # 1. Motion cost
        if self.include_motion and len(T_cam_list) > 0:
            if self.use_cauchy:
                _, _, J_mot = evaluate_motion_cost(
                    T_cam_list,
                    T_lidar_list,
                    R_calib,
                    t_calib,
                    scale,
                    cauchy_c=self.cauchy_param,
                    omega_rot=self.omega_rot,
                    omega_trans=self.omega_trans,
                    use_matrix_form=self.use_matrix_form,
                )
            else:
                # Non-robust L2 cost
                res_r, res_t = compute_motion_residuals(
                    T_cam_list, T_lidar_list, R_calib, t_calib, scale,
                    omega_rot_diag=np.full(9 if self.use_matrix_form else 3, self.omega_rot),
                    omega_trans_diag=np.full(3, self.omega_trans),
                    use_matrix_form=self.use_matrix_form,
                    apply_cauchy_weighting=False,
                )
                J_mot = 0.5 * (np.dot(res_r, res_r) + np.dot(res_t, res_t))
        else:
            J_mot = 0.0

        # 2. Point correspondence cost
        if len(pts_3d) > 0:
            if self.use_cauchy:
                J_corr = evaluate_point_cost(
                    pts_3d,
                    pts_2d,
                    K,
                    T_curr,
                    cauchy_c=self.cauchy_param,
                    omega_corr=self.omega_corr,
                )
            else:
                # Non-robust L2 cost
                res_pts, _ = compute_point_residuals(
                    pts_3d, pts_2d, K, T_curr,
                    omega_corr_diag=np.array([self.omega_corr, self.omega_corr]),
                    apply_cauchy_weighting=False,
                )
                J_corr = 0.5 * np.dot(res_pts, res_pts) if len(res_pts) > 0 else 0.0
        else:
            J_corr = 0.0

        return J_mot, J_corr, J_mot + J_corr

    def optimize(
        self,
        T_calib_init: np.ndarray,
        T_cam_list: List[np.ndarray],
        T_lidar_list: List[np.ndarray],
        pts_3d_lidar: np.ndarray,
        pts_2d_cmr: np.ndarray,
        K: np.ndarray,
        scale: float = 1.0,
    ) -> Tuple[np.ndarray, dict]:
        """
        Run joint optimization.
        
        Args:
            T_calib_init: 4x4 initial calibration transform from coarse registration
            T_cam_list: List of relative camera motion matrices
            T_lidar_list: List of relative LiDAR motion matrices
            pts_3d_lidar: (N, 3) 3D points in LiDAR coordinates
            pts_2d_cmr: (N, 2) 2D pixel coordinates matched by CMRNext
            K: (3, 3) camera intrinsic matrix
            scale: Camera odometry metric scale
            
        Returns:
            T_calib_final: Refined 4x4 calibration transform
            info: Optimization history and diagnostics
        """
        R_cur = T_calib_init[:3, :3].copy()
        t_cur = T_calib_init[:3, 3].copy()

        cost_history = []
        lambda_lm = 1e-2

        init_mot, init_corr, init_total = self.evaluate_total_cost(
            R_cur, t_cur, scale, T_cam_list, T_lidar_list, pts_3d_lidar, pts_2d_cmr, K
        )
        cost_history.append(init_total)

        from mdpcalib.optimization.motion_residuals import extract_pose_arrays, compute_motion_residuals_vec
        has_motion = self.include_motion and len(T_cam_list) > 0
        if has_motion:
            R_cam_all, t_cam_all, R_lid_all, t_lid_all = extract_pose_arrays(T_cam_list, T_lidar_list)

        for iteration in range(self.max_iterations):
            T_curr = make_transform(R_cur, t_cur)
            r_blocks = []
            J_blocks = []

            # --- Motion constraints block ---
            if has_motion:
                res_rot, res_trans = compute_motion_residuals_vec(
                    R_cam_all,
                    t_cam_all,
                    R_lid_all,
                    t_lid_all,
                    R_cur,
                    t_cur,
                    scale,
                    use_matrix_form=self.use_matrix_form,
                    apply_cauchy_weighting=self.use_cauchy,
                    cauchy_c=self.cauchy_param,
                )
                r_mot = np.concatenate([res_rot, res_trans])
                M_mot = len(r_mot)

                # Numerical Jacobian w.r.t [dw (3), dt (3)]
                eps = 1e-6
                J_mot = np.zeros((M_mot, 6), dtype=np.float64)

                # dw (cols 0..2)
                for j in range(3):
                    dw = np.zeros(3)
                    dw[j] = eps
                    R_p = so3_exp(dw) @ R_cur
                    r_rot_p, r_trans_p = compute_motion_residuals_vec(
                        R_cam_all, t_cam_all, R_lid_all, t_lid_all,
                        R_p, t_cur, scale,
                        use_matrix_form=self.use_matrix_form,
                        apply_cauchy_weighting=self.use_cauchy,
                        cauchy_c=self.cauchy_param,
                    )
                    J_mot[:, j] = (np.concatenate([r_rot_p, r_trans_p]) - r_mot) / eps

                # dt (cols 3..5)
                for j in range(3):
                    dt = np.zeros(3)
                    dt[j] = eps
                    t_p = t_cur + dt
                    r_rot_p, r_trans_p = compute_motion_residuals_vec(
                        R_cam_all, t_cam_all, R_lid_all, t_lid_all,
                        R_cur, t_p, scale,
                        use_matrix_form=self.use_matrix_form,
                        apply_cauchy_weighting=self.use_cauchy,
                        cauchy_c=self.cauchy_param,
                    )
                    J_mot[:, 3 + j] = (np.concatenate([r_rot_p, r_trans_p]) - r_mot) / eps

                r_blocks.append(r_mot)
                J_blocks.append(J_mot)

            # --- Point correspondence constraints block ---
            if len(pts_3d_lidar) > 0:
                res_pts, valid_mask = compute_point_residuals(
                    pts_3d_lidar,
                    pts_2d_cmr,
                    K,
                    T_curr,
                    omega_corr_diag=np.array([self.omega_corr, self.omega_corr]),
                    apply_cauchy_weighting=self.use_cauchy,
                    cauchy_c=self.cauchy_param,
                )
                if len(res_pts) > 0:
                    # Analytical Jacobian for point reprojection w.r.t [dt, dw]
                    J_pts_dt_dw, _ = compute_point_jacobian_se3(pts_3d_lidar, K, T_curr)

                    # Permute columns to match [dw (0..2), dt (3..5)]
                    J_pts = np.hstack([J_pts_dt_dw[:, 3:6], J_pts_dt_dw[:, 0:3]])

                    if self.use_cauchy:
                        # Multiply rows by Cauchy weight
                        res_2d = res_pts.reshape((-1, 2))
                        norms = np.linalg.norm(res_2d, axis=1)
                        weights = cauchy_weight(norms, c=self.cauchy_param)
                        W_row = np.repeat(weights, 2)[:, np.newaxis]
                        J_pts = J_pts * W_row

                    r_blocks.append(res_pts)
                    J_blocks.append(J_pts)

            if not r_blocks:
                break

            r_all = np.concatenate(r_blocks)
            J_all = np.vstack(J_blocks)

            # Normal equations
            H = J_all.T @ J_all
            g = -J_all.T @ r_all

            diag_H = np.diag(H)
            H_damped = H + lambda_lm * np.diag(np.maximum(diag_H, 1e-4))

            try:
                delta = np.linalg.solve(H_damped, g)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H_damped, g, rcond=None)[0]

            # delta = [dw_x, dw_y, dw_z, dt_x, dt_y, dt_z]
            dw = delta[:3]
            dt = delta[3:6]

            R_cand = so3_exp(dw) @ R_cur
            t_cand = t_cur + dt

            _, _, cand_cost = self.evaluate_total_cost(
                R_cand, t_cand, scale, T_cam_list, T_lidar_list, pts_3d_lidar, pts_2d_cmr, K
            )

            current_cost = cost_history[-1]
            if cand_cost < current_cost:
                # Accept step
                R_cur = R_cand
                t_cur = t_cand
                cost_history.append(cand_cost)
                lambda_lm = max(1e-7, lambda_lm / 3.0)

                if abs(current_cost - cand_cost) < self.tolerance or np.linalg.norm(delta) < self.tolerance:
                    break
            else:
                # Reject step
                lambda_lm = min(1e7, lambda_lm * 2.0)
                if lambda_lm > 1e6:
                    break

        T_calib_final = make_transform(R_cur, t_cur)
        fin_mot, fin_corr, fin_total = self.evaluate_total_cost(
            R_cur, t_cur, scale, T_cam_list, T_lidar_list, pts_3d_lidar, pts_2d_cmr, K
        )

        info = {
            "iterations": iteration + 1,
            "initial_cost": cost_history[0],
            "final_cost": cost_history[-1],
            "final_motion_cost": fin_mot,
            "final_point_cost": fin_corr,
            "cost_history": cost_history,
        }
        return T_calib_final, info
