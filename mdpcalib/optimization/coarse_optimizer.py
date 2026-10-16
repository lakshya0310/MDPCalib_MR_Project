"""
Coarse Registration Optimizer for MDPCalib.
Solves Eq. (9):
T_init^calib = argmin_{T_calib, s} (J_rot + J_trans) = argmin J_mot
using non-linear least squares with Cauchy robust loss.
"""

from typing import List, Tuple, Optional, Union
import numpy as np
from mdpcalib.geometry.transforms import (
    so3_exp,
    so3_log,
    rotmat_to_quat,
    quat_to_rotmat,
    make_transform,
    invert_transform,
)
from mdpcalib.geometry.hand_eye import solve_hand_eye_shiu_ahmad
from mdpcalib.optimization.robust_losses import cauchy_loss, cauchy_weight
from mdpcalib.optimization.motion_residuals import (
    compute_motion_residuals,
    evaluate_motion_cost,
)


class CoarseOptimizer:
    """
    Solves coarse camera-LiDAR calibration from sensor motion odometry.
    Matches relative camera motion T_cam and relative LiDAR motion T_lidar.
    """

    def __init__(
        self,
        cauchy_param: float = 1e-3,
        omega_rot: float = 1.0,
        omega_trans: float = 1.0,
        max_iterations: int = 150,
        tolerance: float = 1e-8,
        use_matrix_form: bool = True,
        initial_scale: float = 10.0,
    ):
        self.cauchy_param = cauchy_param
        self.omega_rot = omega_rot
        self.omega_trans = omega_trans
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.use_matrix_form = use_matrix_form
        self.initial_scale = initial_scale

    def optimize(
        self,
        T_cam_list: List[np.ndarray],
        T_lidar_list: List[np.ndarray],
        initial_guess: Optional[np.ndarray] = None,
        initial_scale: Optional[float] = None,
        skip_n_poses: int = 0,
    ) -> Tuple[np.ndarray, float, dict]:
        """
        Run coarse registration optimization.
        
        Args:
            T_cam_list: List of relative camera motion transforms (4x4)
            T_lidar_list: List of relative LiDAR motion transforms (4x4)
            initial_guess: Optional 4x4 initial transform T_init (if None, solved via algebraic Shiu-Ahmad)
            initial_scale: Optional VO scale factor (default 10.0 or estimated)
            skip_n_poses: Number of initial poses to skip
            
        Returns:
            T_calib_init: Optimized initial 4x4 calibration transform
            scale: Optimized scale factor
            info: Optimization history and diagnostics
        """
        if skip_n_poses > 0:
            T_cam_list = T_cam_list[skip_n_poses:]
            T_lidar_list = T_lidar_list[skip_n_poses:]

        K = len(T_cam_list)
        if K < 3:
            raise ValueError(f"Need at least 3 motion pairs for coarse calibration, got {K}")

        # Compute algebraic closed-form initialization if not provided
        if initial_guess is None:
            T_shiu, s_est = solve_hand_eye_shiu_ahmad(T_cam_list, T_lidar_list, estimate_scale=True)
            R_init = T_shiu[:3, :3]
            t_init = T_shiu[:3, 3]
            s_init = s_est if (s_est > 0.05 and s_est < 1000.0) else (initial_scale or self.initial_scale)
        else:
            R_init = initial_guess[:3, :3].copy()
            t_init = initial_guess[:3, 3].copy()
            s_init = initial_scale if initial_scale is not None else self.initial_scale

        from mdpcalib.optimization.motion_residuals import extract_pose_arrays, compute_motion_residuals_vec

        R_cam_all, t_cam_all, R_lid_all, t_lid_all = extract_pose_arrays(T_cam_list, T_lidar_list)

        # Parameter vector: x = [w_x, w_y, w_z, t_x, t_y, t_z, s] (7 parameters)
        # with R(w) = exp(w) @ R_init
        w_cur = np.zeros(3, dtype=np.float64)
        t_cur = t_init.copy()
        s_cur = float(s_init)
        R_base = R_init.copy()

        cost_history = []
        lambda_lm = 1e-2

        # Evaluate initial cost
        R_cur = so3_exp(w_cur) @ R_base
        _, _, init_cost = evaluate_motion_cost(
            T_cam_list,
            T_lidar_list,
            R_cur,
            t_cur,
            s_cur,
            cauchy_c=self.cauchy_param,
            omega_rot=self.omega_rot,
            omega_trans=self.omega_trans,
            use_matrix_form=self.use_matrix_form,
        )
        cost_history.append(init_cost)

        for iteration in range(self.max_iterations):
            R_cur = so3_exp(w_cur) @ R_base

            # 1. Compute residuals with Cauchy IRLS weights
            res_rot, res_trans = compute_motion_residuals_vec(
                R_cam_all, t_cam_all, R_lid_all, t_lid_all,
                R_cur, t_cur, s_cur,
                use_matrix_form=self.use_matrix_form,
                apply_cauchy_weighting=True,
                cauchy_c=self.cauchy_param,
            )
            r_vec = np.concatenate([res_rot, res_trans])

            # 2. Compute numerical Jacobian J = d r_vec / d [dw, dt, ds] (M x 7)
            eps = 1e-6
            M = len(r_vec)
            J = np.zeros((M, 7), dtype=np.float64)

            # Columns 0..2: rotation dw
            for j in range(3):
                dw = np.zeros(3)
                dw[j] = eps
                R_pert = so3_exp(dw) @ R_cur
                r_rot_p, r_trans_p = compute_motion_residuals_vec(
                    R_cam_all, t_cam_all, R_lid_all, t_lid_all,
                    R_pert, t_cur, s_cur,
                    use_matrix_form=self.use_matrix_form,
                    apply_cauchy_weighting=True,
                    cauchy_c=self.cauchy_param,
                )
                J[:, j] = (np.concatenate([r_rot_p, r_trans_p]) - r_vec) / eps

            # Columns 3..5: translation dt
            for j in range(3):
                dt = np.zeros(3)
                dt[j] = eps
                t_pert = t_cur + dt
                r_rot_p, r_trans_p = compute_motion_residuals_vec(
                    R_cam_all, t_cam_all, R_lid_all, t_lid_all,
                    R_cur, t_pert, s_cur,
                    use_matrix_form=self.use_matrix_form,
                    apply_cauchy_weighting=True,
                    cauchy_c=self.cauchy_param,
                )
                J[:, 3 + j] = (np.concatenate([r_rot_p, r_trans_p]) - r_vec) / eps

            # Column 6: scale ds
            s_pert = s_cur + eps
            r_rot_p, r_trans_p = compute_motion_residuals_vec(
                R_cam_all, t_cam_all, R_lid_all, t_lid_all,
                R_cur, t_cur, s_pert,
                use_matrix_form=self.use_matrix_form,
                apply_cauchy_weighting=True,
                cauchy_c=self.cauchy_param,
            )
            J[:, 6] = (np.concatenate([r_rot_p, r_trans_p]) - r_vec) / eps

            # 3. Normal equations: (J^T J + lambda I) delta = -J^T r
            H = J.T @ J
            g = -J.T @ r_vec

            # Levenberg-Marquardt step
            diag_H = np.diag(H)
            H_damped = H + lambda_lm * np.diag(np.maximum(diag_H, 1e-4))

            try:
                delta = np.linalg.solve(H_damped, g)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H_damped, g, rcond=None)[0]

            # Candidate update
            w_cand = delta[:3]
            t_cand = t_cur + delta[3:6]
            s_cand = max(1e-4, s_cur + delta[6])
            R_cand = so3_exp(w_cand) @ R_cur

            # Evaluate candidate cost
            _, _, cand_cost = evaluate_motion_cost(
                T_cam_list,
                T_lidar_list,
                R_cand,
                t_cand,
                s_cand,
                cauchy_c=self.cauchy_param,
                omega_rot=self.omega_rot,
                omega_trans=self.omega_trans,
                use_matrix_form=self.use_matrix_form,
            )

            current_cost = cost_history[-1]
            if cand_cost < current_cost:
                # Accept step
                R_base = R_cand
                w_cur = np.zeros(3)
                t_cur = t_cand
                s_cur = s_cand
                cost_history.append(cand_cost)
                lambda_lm = max(1e-7, lambda_lm / 3.0)

                # Check convergence
                if abs(current_cost - cand_cost) < self.tolerance or np.linalg.norm(delta) < self.tolerance:
                    break
            else:
                # Reject step, increase damping
                lambda_lm = min(1e7, lambda_lm * 2.0)
                if lambda_lm > 1e6:
                    break

        T_calib_init = make_transform(R_base, t_cur)
        info = {
            "iterations": iteration + 1,
            "initial_cost": cost_history[0],
            "final_cost": cost_history[-1],
            "cost_history": cost_history,
            "final_scale": s_cur,
        }
        return T_calib_init, s_cur, info
