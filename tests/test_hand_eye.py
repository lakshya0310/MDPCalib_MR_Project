"""
Unit tests for closed-form hand-eye calibration (Shiu-Ahmad and Park-Martin).
"""

import numpy as np
import pytest
from mdpcalib.geometry.transforms import make_transform, euler_to_rotmat, invert_transform
from mdpcalib.geometry.hand_eye import solve_hand_eye_shiu_ahmad, solve_hand_eye_park_martin
from mdpcalib.optimization.metrics import evaluate_calibration


def test_hand_eye_ground_truth_recovery():
    # True calibration transform
    R_gt = euler_to_rotmat(0.05, -0.08, 0.12)
    t_gt = np.array([0.25, -0.15, 0.40])
    T_gt = make_transform(R_gt, t_gt)
    T_gt_inv = invert_transform(T_gt)

    true_scale = 1.25

    # Generate synthetic motions: A_i * X = X * B_i
    T_cam_list = []
    T_lidar_list = []

    rng = np.random.default_rng(123)
    for i in range(25):
        r_step = euler_to_rotmat(
            rng.uniform(-0.1, 0.1),
            rng.uniform(-0.1, 0.1),
            rng.uniform(-0.2, 0.2),
        )
        t_step = rng.uniform(0.1, 1.5, size=3)
        T_cam_true = make_transform(r_step, t_step)

        # Scale applied to camera translation: t_cam = t_true / true_scale
        T_cam_meas = make_transform(r_step, t_step / true_scale)
        # B_i = X^{-1} * A_i * X
        T_lidar_true = T_gt_inv @ T_cam_true @ T_gt

        T_cam_list.append(T_cam_meas)
        T_lidar_list.append(T_lidar_true)

    # Solve via Shiu-Ahmad
    T_shiu, s_shiu = solve_hand_eye_shiu_ahmad(T_cam_list, T_lidar_list, estimate_scale=True)
    m_shiu = evaluate_calibration(T_gt, T_shiu)

    assert m_shiu.E_R_deg < 0.1, f"Rotation error {m_shiu.E_R_deg} is too large"
    assert m_shiu.E_t_cm < 1.0, f"Translation error {m_shiu.E_t_cm} is too large"
    assert np.isclose(s_shiu, true_scale, atol=0.05)

    # Solve via Park-Martin
    T_park, s_park = solve_hand_eye_park_martin(T_cam_list, T_lidar_list, estimate_scale=True)
    m_park = evaluate_calibration(T_gt, T_park)

    assert m_park.E_R_deg < 0.1
    assert m_park.E_t_cm < 1.0
