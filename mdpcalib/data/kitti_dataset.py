"""
KITTI Odometry Dataset loader and calibration file parser.
Supports:
- Sequence 00 (and others)
- Left RGB camera (cam2) and Right RGB camera (cam3)
- Velodyne HDL-64E point cloud parsing (.bin)
- Calibration parameters (P_rect, R_rect, Tr_velo_to_cam)
- Odometry poses and timestamp files
"""

from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
import numpy as np


def load_kitti_poses(poses_file: Union[str, Path]) -> List[np.ndarray]:
    """
    Load KITTI poses from text file (each line has 12 floats representing 3x4 matrix).
    Returns:
    List of 4x4 homogeneous transformation matrices.
    """
    poses = []
    with open(poses_file, "r") as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) == 12:
                T = np.eye(4, dtype=np.float64)
                T[:3, :4] = np.array(vals, dtype=np.float64).reshape((3, 4))
                poses.append(T)
    return poses


def load_kitti_timestamps(times_file: Union[str, Path]) -> np.ndarray:
    """Load timestamps from text file (seconds)."""
    times = []
    with open(times_file, "r") as f:
        for line in f:
            times.append(float(line.strip()))
    return np.array(times, dtype=np.float64)


def load_kitti_point_cloud(bin_file: Union[str, Path]) -> np.ndarray:
    """
    Load Velodyne point cloud from .bin file.
    Returns:
    (N, 4) or (N, 3) float64 array of [x, y, z, reflectance].
    """
    raw = np.fromfile(bin_file, dtype=np.float32)
    return raw.reshape((-1, 4)).astype(np.float64)


def load_kitti_calib(calib_file: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Parse KITTI calibration text file.
    Returns dict mapping key name (e.g. 'P0', 'P1', 'P2', 'P3', 'Tr') to numpy arrays.
    """
    calib = {}
    with open(calib_file, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, val_str = line.split(":", 1)
            key = key.strip()
            vals = np.array([float(x) for x in val_str.strip().split()], dtype=np.float64)
            if key in ["P0", "P1", "P2", "P3", "P_rect_00", "P_rect_01", "P_rect_02", "P_rect_03"]:
                calib[key] = vals.reshape((3, 4))
            elif key in ["R0_rect", "R_rect_00", "R_rect_01", "R_rect_02", "R_rect_03"]:
                calib[key] = vals.reshape((3, 3))
            elif key in ["Tr", "Tr_velo_to_cam", "Tr_velo_cam"]:
                Tr = np.eye(4, dtype=np.float64)
                Tr[:3, :4] = vals.reshape((3, 4))
                calib[key] = Tr
            else:
                calib[key] = vals
    return calib


class KittiOdometryDataset:
    """
    Dataset wrapper for a KITTI sequence.
    """

    def __init__(
        self,
        sequence_dir: Union[str, Path],
        camera: str = "right",  # "left" (cam 2) or "right" (cam 3)
    ):
        self.seq_dir = Path(sequence_dir)
        self.camera = camera.lower()

        self.calib_file = self.seq_dir / "calib.txt"
        self.poses_file = self.seq_dir / "poses.txt"
        self.times_file = self.seq_dir / "times.txt"
        self.velo_dir = self.seq_dir / "velodyne"
        self.image_dir = self.seq_dir / ("image_3" if self.camera == "right" else "image_2")

        self.calib = {}
        if self.calib_file.exists():
            self.calib = load_kitti_calib(self.calib_file)

        self.poses = []
        if self.poses_file.exists():
            self.poses = load_kitti_poses(self.poses_file)

        self.timestamps = np.empty(0)
        if self.times_file.exists():
            self.timestamps = load_kitti_timestamps(self.times_file)

    def get_intrinsics(self) -> np.ndarray:
        """Get 3x3 camera intrinsic matrix K."""
        p_key = "P3" if self.camera == "right" else "P2"
        if p_key in self.calib:
            P = self.calib[p_key]
            return P[:3, :3]
        # Standard KITTI Odometry Seq 00 intrinsics:
        # P2: fx=718.856, fy=718.856, cx=607.1928, cy=185.2157
        return np.array([
            [718.856, 0.0, 607.1928],
            [0.0, 718.856, 185.2157],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def get_ground_truth_transform(self) -> np.ndarray:
        """
        Get 4x4 ground truth extrinsic calibration matrix T_cam_velo.
        """
        tr_key = "Tr" if "Tr" in self.calib else "Tr_velo_to_cam"
        if tr_key in self.calib:
            T_velo_cam0 = self.calib[tr_key]
            R_rect = self.calib.get("R0_rect", np.eye(3))
            T_rect = np.eye(4, dtype=np.float64)
            T_rect[:3, :3] = R_rect
            T_cam0_velo = T_rect @ T_velo_cam0

            if self.camera == "right":
                # Offset for right camera
                P3 = self.calib.get("P3", None)
                if P3 is not None:
                    fx = P3[0, 0]
                    tx = P3[0, 3] / fx  # ~ -0.537m baseline
                    T_cam3_cam0 = np.eye(4, dtype=np.float64)
                    T_cam3_cam0[0, 3] = tx
                    return T_cam3_cam0 @ T_cam0_velo
            return T_cam0_velo

        # Default ground truth for sequence 00 from paper:
        from mdpcalib.data.synthetic_data import get_kitti_ground_truth
        return get_kitti_ground_truth(camera=self.camera)
