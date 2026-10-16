"""
Unit tests for residuals, Cauchy loss, and Jacobians.
"""

import numpy as np
import pytest
from mdpcalib.geometry.transforms import make_transform, euler_to_rotmat, project_points
from mdpcalib.optimization.robust_losses import cauchy_loss, cauchy_derivatives, cauchy_weight
from mdpcalib.optimization.motion_residuals import (
    compute_rotation_residual,
    compute_translation_residual,
    compute_motion_residuals,
)
from mdpcalib.optimization.point_residuals import (
    compute_point_residual,
    compute_point_residuals,
    compute_point_jacobian_se3,
)


def test_cauchy_loss_properties():
    # At s = 0, rho(0) = 0
    assert cauchy_loss(0.0, c=1.0) == 0.0
    # For small s, rho(s) ~ s
    assert np.isclose(cauchy_loss(1e-4, c=1.0), 1e-4, atol=1e-6)
    # Derivative at s=0 is 1.0
    rho_p, rho_pp = cauchy_derivatives(0.0, c=1.0)
    assert np.isclose(rho_p, 1.0)
    # Weight at r=0 is 1.0
    assert np.isclose(cauchy_weight(0.0, c=1.0), 1.0)
    # Weight at large r is dampened
    assert cauchy_weight(100.0, c=1.0) < 0.02


def test_motion_residuals_zero_at_truth():
    R_gt = euler_to_rotmat(0.1, -0.2, 0.3)
    t_gt = np.array([0.5, -0.4, 0.3])
    T_gt = make_transform(R_gt, t_gt)

    R_cam = euler_to_rotmat(0.02, 0.01, -0.03)
    t_cam = np.array([0.05, 0.02, 1.0])
    T_cam = make_transform(R_cam, t_cam)

    # Exact T_lidar = T_gt^{-1} * T_cam * T_gt
    from mdpcalib.geometry.transforms import invert_transform
    T_lidar = invert_transform(T_gt) @ T_cam @ T_gt

    # Evaluated at true calibration with scale s=1.0
    e_rot = compute_rotation_residual(R_cam, T_lidar[:3, :3], R_gt, use_matrix_form=True)
    assert np.allclose(e_rot, 0.0, atol=1e-8)

    e_trans = compute_translation_residual(
        R_cam, t_cam, T_lidar[:3, :3], T_lidar[:3, 3], R_gt, t_gt, scale=1.0
    )
    assert np.allclose(e_trans, 0.0, atol=1e-8)


def test_point_analytical_jacobian():
    # Verify analytical Jacobian of 2D reprojection w.r.t SE(3) perturbation
    K = np.array([[700.0, 0.0, 600.0], [0.0, 700.0, 200.0], [0.0, 0.0, 1.0]])
    R = euler_to_rotmat(0.05, -0.02, 0.03)
    t = np.array([0.1, -0.2, 0.3])
    T = make_transform(R, t)

    pts_3d = np.array([
        [1.5, 2.0, 15.0],
        [-2.0, 1.0, 25.0],
    ])

    J_ana, valid = compute_point_jacobian_se3(pts_3d, K, T)
    assert np.all(valid)
    assert J_ana.shape == (4, 6)

    # Numerical Jacobian check for point 0
    eps = 1e-6
    from mdpcalib.geometry.transforms import se3_exp
    uv_base, _ = project_points(pts_3d, K, T)

    J_num = np.zeros((4, 6))
    for j in range(6):
        xi = np.zeros(6)
        xi[j] = eps
        T_pert = se3_exp(xi) @ T
        uv_pert, _ = project_points(pts_3d, K, T_pert)
        diff = (uv_pert - uv_base) / eps
        J_num[0::2, j] = diff[:, 0]
        J_num[1::2, j] = diff[:, 1]

    assert np.allclose(J_ana, J_num, atol=1e-4)
