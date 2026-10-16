"""
Geometric transformation utilities:
- Quaternions [x, y, z, w]
- SO(3) and SE(3) Lie group / Lie algebra operations
- Euler angles (ZYX: yaw, pitch, roll)
- Rigid body transforms and Pinhole camera projections
"""

from typing import Tuple, Union
import numpy as np


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion [x, y, z, w]."""
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q_norm = q / norm
    if q_norm[3] < 0:
        q_norm = -q_norm  # Enforce positive scalar part
    return q_norm


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions q1 * q2, format [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64)


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Inverse of unit quaternion [x, y, z, w]."""
    q = normalize_quat(q)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_distance(q1: np.ndarray, q2: np.ndarray, geodesic: bool = True) -> float:
    """
    Compute rotation distance between two quaternions in radians.
    If geodesic=True: 2 * atan2(||r_vec||, |r_w|) as used in Optimizer::GetQuatDist.
    If geodesic=False: atan2(||r_vec||, r_w) matching paper Eq. (14c).
    """
    q1_n = normalize_quat(q1)
    q2_n = normalize_quat(q2)
    r = quat_multiply(q1_n, quat_inverse(q2_n))
    vec_norm = np.linalg.norm(r[:3])
    w = np.abs(r[3]) if geodesic else r[3]
    angle = np.arctan2(vec_norm, w)
    return float(2.0 * angle if geodesic else angle)


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion [x, y, z, w]."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return normalize_quat(np.array([x, y, z, w], dtype=np.float64))


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [x, y, z, w] to 3x3 rotation matrix."""
    q = normalize_quat(q)
    x, y, z, w = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Return 3x3 skew symmetric matrix for vector v in R^3."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Exponential map from so(3) vector in R^3 to SO(3) 3x3 rotation matrix (Rodrigues formula)."""
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64)
    K = skew_symmetric(phi / theta)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithmic map from SO(3) 3x3 rotation matrix to so(3) vector in R^3."""
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-10:
        return np.zeros(3, dtype=np.float64)
    if np.pi - theta < 1e-5:
        # Near pi ambiguity
        diag = np.maximum(0.0, np.diag(R) + 1.0)
        k = np.sqrt(diag / 2.0)
        return theta * k
    sin_theta = np.sin(theta)
    K = (R - R.T) / (2.0 * sin_theta)
    return np.array([K[2, 1], K[0, 2], K[1, 0]], dtype=np.float64) * theta


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """
    Exponential map from se(3) tangent vector xi = [v, omega] in R^6 to SE(3) 4x4 matrix.
    v in R^3 (linear translation part), omega in R^3 (rotation vector).
    """
    v = xi[:3]
    omega = xi[3:]
    theta = np.linalg.norm(omega)
    T = np.eye(4, dtype=np.float64)
    if theta < 1e-10:
        T[:3, :3] = np.eye(3)
        T[:3, 3] = v
        return T

    K = skew_symmetric(omega / theta)
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    V = np.eye(3) + ((1.0 - np.cos(theta)) / (theta)) * K + ((theta - np.sin(theta)) / (theta)) * (K @ K)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """Logarithmic map from SE(3) 4x4 matrix to se(3) vector in R^6."""
    R = T[:3, :3]
    t = T[:3, 3]
    omega = so3_log(R)
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.concatenate([t, omega])
    K = skew_symmetric(omega / theta)
    V_inv = np.eye(3) - 0.5 * theta * K + (1.0 - (theta * np.sin(theta)) / (2.0 * (1.0 - np.cos(theta)))) * (K @ K)
    v = V_inv @ t
    return np.concatenate([v, omega])


def rotmat_to_euler(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to Euler angles [roll, pitch, yaw] in radians.
    Convention: ZYX (yaw around Z, pitch around Y, roll around X).
    """
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

    if np.abs(r31) < 1.0 - 1e-10:
        pitch = -np.arcsin(r31)
        roll = np.arctan2(r32, r33)
        yaw = np.arctan2(r21, r11)
    else:
        # Gimbal lock
        yaw = 0.0
        if r31 <= -1.0:
            pitch = np.pi / 2.0
            roll = yaw + np.arctan2(r12, r13)
        else:
            pitch = -np.pi / 2.0
            roll = -yaw + np.arctan2(-r12, -r13)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def euler_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles [roll, pitch, yaw] in radians to 3x3 rotation matrix (ZYX order)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Create 4x4 homogeneous transformation matrix from R (3x3) and t (3,)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert 4x4 rigid body transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def project_points(pts_3d: np.ndarray, K: np.ndarray, T_calib: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D LiDAR points into camera image plane.
    pts_3d: (N, 3) in LiDAR coordinates
    K: (3, 3) camera intrinsic matrix
    T_calib: (4, 4) calibration transform from LiDAR to Camera frame: p_cam = T_calib * p_lidar
    
    Returns:
    uv: (M, 2) pixel coordinates for valid points in front of camera (Z > 0)
    valid_mask: (N,) boolean mask of valid projections
    """
    N = pts_3d.shape[0]
    if N == 0:
        return np.empty((0, 2), dtype=np.float64), np.zeros(0, dtype=bool)

    pts_hom = np.hstack([pts_3d[:, :3], np.ones((N, 1), dtype=np.float64)])  # (N, 4)
    pts_cam = (T_calib @ pts_hom.T).T[:, :3]  # (N, 3)

    z = pts_cam[:, 2]
    valid_mask = z > 0.1  # points in front of camera

    uv = np.zeros((N, 2), dtype=np.float64)
    uv[valid_mask, 0] = (K[0, 0] * pts_cam[valid_mask, 0] / z[valid_mask]) + K[0, 2]
    uv[valid_mask, 1] = (K[1, 1] * pts_cam[valid_mask, 1] / z[valid_mask]) + K[1, 2]

    return uv, valid_mask
