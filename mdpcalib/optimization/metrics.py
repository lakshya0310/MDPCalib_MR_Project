r"""
Evaluation metrics for Camera-LiDAR calibration.
Implements:
- Eq. (14a): Translation error magnitude E_t = ||t - \hat{t}||_2 in cm
- Eq. (14b): Relative quaternion m = q * \hat{q}^{-1}
- Eq. (14c): Rotation error magnitude E_R = atan2(sqrt(m_x^2 + m_y^2 + m_z^2), m_w) in degrees
- Individual axis translation errors (x, y, z) in cm
- Individual axis rotation errors (roll, pitch, yaw) in degrees
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
import numpy as np
from mdpcalib.geometry.transforms import (
    rotmat_to_quat,
    quat_multiply,
    quat_inverse,
    rotmat_to_euler,
    normalize_quat,
)


@dataclass
class CalibrationMetrics:
    # Magnitudes
    E_t_cm: float  # Translation magnitude error in cm
    E_R_deg: float  # Rotation magnitude error in deg (Eq. 14c)
    E_R_geodesic_deg: float  # Standard geodesic angular distance 2*atan2(||m||, |w|) in deg

    # Individual translation errors in cm
    t_x_err_cm: float
    t_y_err_cm: float
    t_z_err_cm: float

    # Individual rotation Euler angle errors in degrees
    roll_err_deg: float
    pitch_err_deg: float
    yaw_err_deg: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def summary_str(self) -> str:
        return (
            f"E_t: {self.E_t_cm:.2f} cm | E_R: {self.E_R_deg:.2f}° (geodesic: {self.E_R_geodesic_deg:.2f}°)\n"
            f"Translation [x, y, z]: [{self.t_x_err_cm:.2f}, {self.t_y_err_cm:.2f}, {self.t_z_err_cm:.2f}] cm\n"
            f"Rotation [roll, pitch, yaw]: [{self.roll_err_deg:.2f}°, {self.pitch_err_deg:.2f}°, {self.yaw_err_deg:.2f}°]"
        )


def evaluate_calibration(
    T_gt: np.ndarray,
    T_pred: np.ndarray,
) -> CalibrationMetrics:
    """
    Compute calibration error metrics against ground truth.
    
    Args:
        T_gt: 4x4 ground truth extrinsic calibration matrix (LiDAR to camera)
        T_pred: 4x4 predicted calibration matrix (LiDAR to camera)
        
    Returns:
        CalibrationMetrics instance containing all metrics in cm and degrees.
    """
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]

    R_pred = T_pred[:3, :3]
    t_pred = T_pred[:3, 3]

    # Eq. (14a): Translation error in cm
    # (assuming t is in meters, convert to cm by multiplying by 100)
    delta_t = t_gt - t_pred
    E_t_meters = np.linalg.norm(delta_t)
    E_t_cm = float(E_t_meters * 100.0)

    t_x_cm = float(abs(delta_t[0]) * 100.0)
    t_y_cm = float(abs(delta_t[1]) * 100.0)
    t_z_cm = float(abs(delta_t[2]) * 100.0)

    # Eq. (14b): Quaternions m = q * \hat{q}^{-1}
    q_gt = rotmat_to_quat(R_gt)
    q_pred = rotmat_to_quat(R_pred)

    m = quat_multiply(q_gt, quat_inverse(q_pred))
    m = normalize_quat(m)
    m_vec_norm = np.linalg.norm(m[:3])
    m_w = m[3]

    # Eq. (14c): Rotation error in degrees
    E_R_rad = np.arctan2(m_vec_norm, m_w)
    E_R_deg = float(np.degrees(E_R_rad))

    # Geodesic angular distance
    E_R_geo_rad = 2.0 * np.arctan2(m_vec_norm, abs(m_w))
    E_R_geo_deg = float(np.degrees(E_R_geo_rad))

    # Euler angles error in degrees (ZYX: roll, pitch, yaw)
    euler_gt = np.degrees(rotmat_to_euler(R_gt))
    euler_pred = np.degrees(rotmat_to_euler(R_pred))
    delta_euler = np.abs(euler_gt - euler_pred)

    # Handle 360 wrap-around
    delta_euler = np.where(delta_euler > 180.0, 360.0 - delta_euler, delta_euler)

    return CalibrationMetrics(
        E_t_cm=E_t_cm,
        E_R_deg=E_R_deg,
        E_R_geodesic_deg=E_R_geo_deg,
        t_x_err_cm=t_x_cm,
        t_y_err_cm=t_y_cm,
        t_z_err_cm=t_z_cm,
        roll_err_deg=float(delta_euler[0]),
        pitch_err_deg=float(delta_euler[1]),
        yaw_err_deg=float(delta_euler[2]),
    )
