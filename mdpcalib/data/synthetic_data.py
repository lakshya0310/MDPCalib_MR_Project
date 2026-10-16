"""
Synthetic trajectory and benchmark data generator for MDPCalib.
Provides:
- Exact KITTI Sequence 00 ground truth and author calibration parameters
- Realistic 6-DoF vehicle odometry trajectory generator matching KITTI dynamics
- Simulated 3D point cloud and 2D correspondences with controllable outlier ratio
- Reproducible benchmarking without needing 100GB+ raw rosbag downloads
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from mdpcalib.geometry.transforms import (
    make_transform,
    invert_transform,
    euler_to_rotmat,
    rotmat_to_euler,
    project_points,
)


def get_kitti_published_parameters(camera: str = "right") -> np.ndarray:
    """
    Returns the exact calibration matrix published by the authors in Supplementary Section S-II.
    """
    if camera.lower() == "left":
        R = np.array([
            [-1.2619e-4, -9.9997e-1, -8.2230e-3],
            [-7.8537e-3,  8.2238e-3, -9.9994e-1],
            [ 9.9997e-1, -6.1602e-5, -7.8545e-3],
        ], dtype=np.float64)
        t = np.array([5.1090e-2, -5.5873e-2, -2.9575e-1], dtype=np.float64)
    else:  # right
        R = np.array([
            [-1.8202e-3, -9.9996e-1, -8.2416e-3],
            [-8.2823e-3,  8.2564e-3, -9.9993e-1],
            [ 9.9996e-1, -1.7518e-3, -8.2970e-3],
        ], dtype=np.float64)
        t = np.array([-4.5166e-1, -4.8448e-2, -2.8787e-1], dtype=np.float64)

    return make_transform(R, t)


def get_kitti_ground_truth(camera: str = "right") -> np.ndarray:
    """
    Returns the ground truth extrinsic calibration matrix T_cam_velo for KITTI Sequence 00.
    Standard Velodyne-to-camera matrix from KITTI benchmark calib.
    """
    # Standard KITTI Sequence 00 Tr_velo_to_cam (to Cam 0):
    R_velo_cam0 = np.array([
        [7.533745e-03, -9.999714e-01, -6.166020e-04],
        [1.480249e-02,  7.280733e-04, -9.998902e-01],
        [9.998621e-01,  7.523790e-03,  1.480755e-02],
    ], dtype=np.float64)
    t_velo_cam0 = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01], dtype=np.float64)

    # R0_rect
    R_rect0 = np.array([
        [9.999239e-01, 9.837760e-03, -7.445048e-03],
        [-9.869795e-03, 9.999421e-01, -4.278459e-03],
        [7.402527e-03, 4.351614e-03, 9.999631e-01],
    ], dtype=np.float64)

    T_rect = np.eye(4, dtype=np.float64)
    T_rect[:3, :3] = R_rect0
    T_velo_cam0_mat = make_transform(R_velo_cam0, t_velo_cam0)
    T_cam0_velo = T_rect @ T_velo_cam0_mat

    if camera.lower() == "right":
        # Baseline between left and right color camera: ~ -0.537m in x
        T_cam3_cam0 = np.eye(4, dtype=np.float64)
        T_cam3_cam0[0, 3] = -0.5371657
        return T_cam3_cam0 @ T_cam0_velo
    elif camera.lower() == "left":
        # Left color camera (cam 2)
        T_cam2_cam0 = np.eye(4, dtype=np.float64)
        T_cam2_cam0[0, 3] = 0.059757
        return T_cam2_cam0 @ T_cam0_velo

    return T_cam0_velo


def generate_synthetic_calibration_data(
    num_poses: int = 1000,
    num_pairs: int = 100,
    points_per_pair: int = 1000,
    camera: str = "right",
    vo_scale: float = 0.85,  # simulated monocular visual odometry scale ambiguity
    motion_noise_rot: float = 0.001,  # rad
    motion_noise_trans: float = 0.01,  # meters
    point_noise: float = 0.5,  # pixels
    outlier_ratio: float = 0.05,  # 5% gross outliers to test Cauchy loss
    seed: int = 42,
) -> Dict[str, any]:
    """
    Generate realistic, time-synchronized odometry trajectories and point correspondences
    matching the KITTI setup in the MDPCalib paper.
    
    Args:
        num_poses: Number of motion odometry poses (paper default: 1000)
        num_pairs: Number of image-point cloud pairs (paper default: 100)
        points_per_pair: Sampled point correspondence count per pair (paper default: ~1000)
        camera: "left" or "right"
        vo_scale: Unobserved monocular scale factor s
        outlier_ratio: Proportion of gross outliers (testing Cauchy robustifier)
        seed: Random seed
        
    Returns:
        dict containing T_cam_list, T_lidar_list, pts_3d, pts_2d, K, T_gt
    """
    rng = np.random.default_rng(seed)
    T_gt = get_kitti_ground_truth(camera=camera)
    T_gt_inv = invert_transform(T_gt)

    # Standard KITTI Camera Intrinsics
    K = np.array([
        [718.856, 0.0, 607.1928],
        [0.0, 718.856, 185.2157],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Generate smooth 6-DoF vehicle path (curves, turns, forward velocity)
    dt = 0.1  # 10 Hz
    t_vals = np.linspace(0, num_poses * dt, num_poses)

    # Forward velocity ~ 10 m/s with speed variations
    v = 10.0 + 1.5 * np.sin(0.1 * t_vals)
    # Yaw rate: gentle curves and turns
    yaw_rate = 0.05 * np.sin(0.08 * t_vals) + 0.02 * np.cos(0.03 * t_vals)

    T_cam_list = []
    T_lidar_list = []

    for i in range(num_poses):
        d_dist = 1.0 + 0.2 * np.sin(0.1 * i)
        d_yaw = 0.04 * np.sin(0.05 * i) + 0.02 * np.cos(0.02 * i)
        d_pitch = 0.025 * np.cos(0.07 * i) + 0.015 * np.sin(0.13 * i)
        d_roll = 0.02 * np.sin(0.11 * i) + 0.01 * np.cos(0.04 * i)

        # Relative vehicle motion in vehicle/camera frame
        R_step_cam = euler_to_rotmat(d_roll, d_pitch, d_yaw)
        t_step_cam = np.array([0.05 * np.sin(0.1 * i), 0.02 * np.cos(0.1 * i), d_dist], dtype=np.float64)

        # Scale ambiguity for monocular visual odometry: t_cam = t_true / s
        t_cam_meas = t_step_cam / vo_scale

        # Add small sensory noise to camera odometry
        noise_r_cam = rng.normal(0, motion_noise_rot, size=3)
        R_cam_noisy = euler_to_rotmat(*noise_r_cam) @ R_step_cam
        t_cam_noisy = t_cam_meas + rng.normal(0, motion_noise_trans / vo_scale, size=3)

        T_cam_i = make_transform(R_cam_noisy, t_cam_noisy)

        # Exact corresponding LiDAR motion:
        # T_cam * T_calib = T_calib * T_lidar => T_lidar = T_calib^{-1} * T_cam_true * T_calib
        T_step_true = make_transform(R_step_cam, t_step_cam)
        T_lidar_true = T_gt_inv @ T_step_true @ T_gt

        # Add small sensory noise to LiDAR odometry
        noise_r_lid = rng.normal(0, motion_noise_rot, size=3)
        R_lid_noisy = euler_to_rotmat(*noise_r_lid) @ T_lidar_true[:3, :3]
        t_lid_noisy = T_lidar_true[:3, 3] + rng.normal(0, motion_noise_trans, size=3)

        T_lidar_i = make_transform(R_lid_noisy, t_lid_noisy)

        T_cam_list.append(T_cam_i)
        T_lidar_list.append(T_lidar_i)

    # Generate 3D point cloud and 2D correspondences
    all_pts_3d = []
    all_pts_2d = []

    # Choose num_pairs evenly distributed along sequence
    pair_indices = np.linspace(0, num_poses - 1, num_pairs, dtype=int)

    for p_idx in pair_indices:
        # Synthesize 3D LiDAR scene points in front of vehicle (depth 2m to 70m)
        num_raw = int(points_per_pair * 1.5)
        depths = rng.uniform(2.0, 70.0, size=num_raw)
        x_pts = rng.uniform(-15.0, 15.0, size=num_raw) * (depths / 30.0)
        y_pts = rng.uniform(-2.0, 5.0, size=num_raw)
        raw_pts_3d_cam = np.stack([x_pts, y_pts, depths], axis=1)

        # Transform to LiDAR frame: p_lidar = T_gt^{-1} * p_cam
        raw_pts_3d_lidar = (T_gt_inv[:3, :3] @ raw_pts_3d_cam.T).T + T_gt_inv[:3, 3]

        # Project via ground truth to get true 2D pixel coordinates
        uv_true, valid = project_points(raw_pts_3d_lidar, K, T_gt)
        valid_3d = raw_pts_3d_lidar[valid]
        valid_2d = uv_true[valid]

        # Keep points within 1242x375 image bounds
        in_bounds = (
            (valid_2d[:, 0] >= 5.0) & (valid_2d[:, 0] <= 1235.0) &
            (valid_2d[:, 1] >= 5.0) & (valid_2d[:, 1] <= 370.0)
        )
        valid_3d = valid_3d[in_bounds]
        valid_2d = valid_2d[in_bounds]

        if len(valid_3d) > points_per_pair:
            sel = rng.choice(len(valid_3d), size=points_per_pair, replace=False)
            valid_3d = valid_3d[sel]
            valid_2d = valid_2d[sel]

        # Add Gaussian matching noise (sub-pixel accuracy of CMRNext ~ 0.5 px)
        noise = rng.normal(0, point_noise, size=valid_2d.shape)
        meas_2d = valid_2d + noise

        # Inject gross outliers for Cauchy robustifier test
        if outlier_ratio > 0.0:
            num_outliers = int(len(meas_2d) * outlier_ratio)
            if num_outliers > 0:
                outlier_idx = rng.choice(len(meas_2d), size=num_outliers, replace=False)
                meas_2d[outlier_idx, 0] += rng.uniform(-80.0, 80.0, size=num_outliers)
                meas_2d[outlier_idx, 1] += rng.uniform(-50.0, 50.0, size=num_outliers)

        all_pts_3d.append(valid_3d)
        all_pts_2d.append(meas_2d)

    pts_3d_concat = np.vstack(all_pts_3d)
    pts_2d_concat = np.vstack(all_pts_2d)

    return {
        "T_cam_list": T_cam_list,
        "T_lidar_list": T_lidar_list,
        "pts_3d_lidar": pts_3d_concat,
        "pts_2d_cmr": pts_2d_concat,
        "K": K,
        "T_gt": T_gt,
        "true_scale": vo_scale,
    }
