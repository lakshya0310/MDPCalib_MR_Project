"""
Unit tests for mdpcalib.geometry:
- Quaternion algebra
- SO(3) and SE(3) Lie group maps
- Euler angles conversion
- SLERP pose interpolation
- Pinhole camera projection
"""

import numpy as np
import pytest
from mdpcalib.geometry.transforms import (
    normalize_quat,
    quat_multiply,
    quat_inverse,
    quat_distance,
    rotmat_to_quat,
    quat_to_rotmat,
    rotmat_to_euler,
    euler_to_rotmat,
    so3_exp,
    so3_log,
    se3_exp,
    se3_log,
    make_transform,
    invert_transform,
    project_points,
)
from mdpcalib.geometry.synchronization import slerp, interpolate_pose


def test_quaternion_identity():
    q_id = np.array([0.0, 0.0, 0.0, 1.0])
    R_id = quat_to_rotmat(q_id)
    assert np.allclose(R_id, np.eye(3))
    q_rec = rotmat_to_quat(R_id)
    assert np.allclose(q_rec, q_id)


def test_quaternion_inverse_multiply():
    q = normalize_quat(np.array([0.2, -0.3, 0.4, 0.8]))
    q_inv = quat_inverse(q)
    prod = quat_multiply(q, q_inv)
    assert np.allclose(prod, np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-7)


def test_rotmat_quat_roundtrip():
    angles = np.radians([15.0, -25.0, 40.0])
    R = euler_to_rotmat(*angles)
    q = rotmat_to_quat(R)
    R_rec = quat_to_rotmat(q)
    assert np.allclose(R, R_rec, atol=1e-6)

    euler_rec = rotmat_to_euler(R)
    assert np.allclose(angles, euler_rec, atol=1e-6)


def test_so3_exp_log():
    w = np.array([0.05, -0.12, 0.08])
    R = so3_exp(w)
    w_rec = so3_log(R)
    assert np.allclose(w, w_rec, atol=1e-6)


def test_se3_exp_log():
    xi = np.array([1.2, -0.5, 3.4, 0.03, -0.04, 0.02])
    T = se3_exp(xi)
    xi_rec = se3_log(T)
    assert np.allclose(xi, xi_rec, atol=1e-5)


def test_invert_transform():
    R = euler_to_rotmat(0.1, -0.2, 0.3)
    t = np.array([1.0, 2.0, 3.0])
    T = make_transform(R, t)
    T_inv = invert_transform(T)
    assert np.allclose(T @ T_inv, np.eye(4), atol=1e-7)


def test_slerp_interpolation():
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = rotmat_to_quat(euler_to_rotmat(0.0, 0.0, np.radians(90.0)))

    # Midpoint should be 45 degrees
    q_mid = slerp(q0, q1, 0.5)
    R_mid = quat_to_rotmat(q_mid)
    euler_mid = rotmat_to_euler(R_mid)
    assert np.isclose(euler_mid[2], np.radians(45.0), atol=1e-4)


def test_project_points():
    K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 300.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    # Point at (X=1, Y=2, Z=10)
    pts = np.array([[1.0, 2.0, 10.0]])
    uv, valid = project_points(pts, K, T)
    assert valid[0]
    expected_u = 1000.0 * (1.0 / 10.0) + 500.0  # 600.0
    expected_v = 1000.0 * (2.0 / 10.0) + 300.0  # 500.0
    assert np.isclose(uv[0, 0], expected_u)
    assert np.isclose(uv[0, 1], expected_v)
