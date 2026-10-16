"""
Geometry subpackage for coordinate transformations, projections, and hand-eye kinematics.
"""

from mdpcalib.geometry.transforms import (
    rotmat_to_quat,
    quat_to_rotmat,
    quat_multiply,
    quat_inverse,
    quat_distance,
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
from mdpcalib.geometry.synchronization import interpolate_pose, synchronize_trajectories
from mdpcalib.geometry.hand_eye import solve_hand_eye_shiu_ahmad, solve_hand_eye_park_martin

__all__ = [
    "rotmat_to_quat",
    "quat_to_rotmat",
    "quat_multiply",
    "quat_inverse",
    "quat_distance",
    "rotmat_to_euler",
    "euler_to_rotmat",
    "so3_exp",
    "so3_log",
    "se3_exp",
    "se3_log",
    "make_transform",
    "invert_transform",
    "project_points",
    "interpolate_pose",
    "synchronize_trajectories",
    "solve_hand_eye_shiu_ahmad",
    "solve_hand_eye_park_martin",
]
