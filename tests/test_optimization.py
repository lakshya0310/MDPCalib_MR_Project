"""
Unit tests for coarse and joint optimizers.
"""

import numpy as np
import pytest
from mdpcalib.data.synthetic_data import generate_synthetic_calibration_data
from mdpcalib.optimization.coarse_optimizer import CoarseOptimizer
from mdpcalib.optimization.joint_optimizer import JointOptimizer
from mdpcalib.optimization.metrics import evaluate_calibration


def test_coarse_optimizer_convergence():
    data = generate_synthetic_calibration_data(
        num_poses=100,
        num_pairs=20,
        camera="right",
        seed=42,
    )

    opt = CoarseOptimizer(max_iterations=50)
    T_coarse, scale, info = opt.optimize(
        T_cam_list=data["T_cam_list"],
        T_lidar_list=data["T_lidar_list"],
        skip_n_poses=5,
    )

    m = evaluate_calibration(data["T_gt"], T_coarse)
    assert m.E_R_deg < 1.0, f"Coarse rotation error {m.E_R_deg}° should be sub-degree"
    assert info["final_cost"] <= info["initial_cost"]


def test_joint_optimizer_refinement():
    data = generate_synthetic_calibration_data(
        num_poses=100,
        num_pairs=25,
        camera="right",
        seed=42,
    )

    coarse_opt = CoarseOptimizer(max_iterations=50)
    T_coarse, scale, _ = coarse_opt.optimize(
        T_cam_list=data["T_cam_list"],
        T_lidar_list=data["T_lidar_list"],
        skip_n_poses=5,
    )
    m_coarse = evaluate_calibration(data["T_gt"], T_coarse)

    joint_opt = JointOptimizer(max_iterations=50)
    T_final, info = joint_opt.optimize(
        T_calib_init=T_coarse,
        T_cam_list=data["T_cam_list"],
        T_lidar_list=data["T_lidar_list"],
        pts_3d_lidar=data["pts_3d_lidar"],
        pts_2d_cmr=data["pts_2d_cmr"],
        K=data["K"],
        scale=scale,
    )
    m_final = evaluate_calibration(data["T_gt"], T_final)

    # Fine joint optimization should substantially improve calibration accuracy
    assert m_final.E_t_cm < m_coarse.E_t_cm, "Joint optimization must reduce translation error"
    assert m_final.E_R_deg <= m_coarse.E_R_deg + 0.05
