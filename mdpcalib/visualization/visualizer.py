"""
Visualization tools for camera-LiDAR calibration:
- LiDAR point cloud overlay onto images colored by depth (matching Fig. 10)
- Cost convergence curves
- Reproduction of Ablation study plots (Figs. 6, 7, 8, 9)
"""

from typing import Optional, Union, List, Dict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / server execution
import matplotlib.pyplot as plt
from mdpcalib.geometry.transforms import project_points


def project_points_to_image(
    pts_3d: np.ndarray,
    img: Optional[np.ndarray],
    K: np.ndarray,
    T_calib: np.ndarray,
    img_width: int = 1242,
    img_height: int = 375,
    point_size: int = 2,
    max_depth: float = 80.0,
    save_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Project LiDAR point cloud onto image and color by depth (jet colormap), matching Figure 10.
    
    Args:
        pts_3d: (N, 3) in LiDAR coordinates
        img: (H, W, 3) RGB image or None (creates dark canvas)
        K: (3, 3) intrinsic matrix
        T_calib: (4, 4) calibration transform
        img_width: Canvas width
        img_height: Canvas height
        point_size: Radius of projected points
        max_depth: Depth normalization upper bound (meters)
        save_path: Output image path
        
    Returns:
        canvas: (H, W, 3) uint8 image with LiDAR projection
    """
    if img is not None:
        canvas = img.copy()
        h, w = canvas.shape[:2]
    else:
        h, w = img_height, img_width
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    uv, valid = project_points(pts_3d, K, T_calib)
    valid_pts = pts_3d[valid]
    valid_uv = uv[valid]

    # Transform to camera frame to get actual depth Z
    pts_cam = (T_calib[:3, :3] @ valid_pts.T).T + T_calib[:3, 3]
    depths = pts_cam[:, 2]

    # In-bounds mask
    in_bounds = (
        (valid_uv[:, 0] >= 0) & (valid_uv[:, 0] < w) &
        (valid_uv[:, 1] >= 0) & (valid_uv[:, 1] < h) &
        (depths > 0.5) & (depths <= max_depth)
    )

    valid_uv = valid_uv[in_bounds].astype(int)
    depths = depths[in_bounds]

    # Normalized depth for colormap
    norm_depth = np.clip(depths / max_depth, 0.0, 1.0)
    cmap = plt.get_cmap("turbo")
    colors = (cmap(norm_depth)[:, :3] * 255).astype(np.uint8)

    # Draw points on canvas
    for i in range(len(valid_uv)):
        u, v = valid_uv[i]
        c = colors[i]
        r = point_size
        canvas[max(0, v - r):min(h, v + r + 1), max(0, u - r):min(w, u + r + 1)] = c

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 4), dpi=150)
        plt.imshow(canvas)
        plt.axis("off")
        plt.title(f"Camera-LiDAR Projection (Colored by depth up to {max_depth:.0f}m)")
        plt.tight_layout()
        plt.savefig(p, bbox_inches="tight")
        plt.close()

    return canvas


def plot_convergence(
    coarse_history: List[float],
    joint_history: List[float],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot optimization cost convergence across coarse and joint stages.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

    # Coarse registration
    ax1.plot(coarse_history, marker="o", color="#1f77b4", linewidth=2, markersize=4)
    ax1.set_title("Coarse Registration (Motion Constraints $J_{mot}$)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Cauchy Cost")
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Joint registration
    ax2.plot(joint_history, marker="s", color="#2ca02c", linewidth=2, markersize=4)
    ax2.set_title("Fine Registration (Joint $J_{mot} + J_{corr}$)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Joint Cauchy Cost")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight")
    plt.close()


def plot_ablation_curves(
    results_dict: Dict[str, any],
    save_dir: Union[str, Path],
) -> None:
    """
    Generate ablation study plots matching Figures 6, 7, 8, 9 of the paper:
    - Fig 6: Errors vs Number of Poses (200 to 1000)
    - Fig 7: Errors vs Relative Number of Correspondences (1% to 5%)
    - Fig 8: Errors vs Number of Image-PCL Pairs (20 to 100)
    - Fig 9: Pair Diversity vs Relative Correspondences
    """
    save_p = Path(save_dir)
    save_p.mkdir(parents=True, exist_ok=True)

    # Fig 6: Number of Poses
    if "poses_study" in results_dict:
        data = results_dict["poses_study"]
        poses = data["x"]
        et = data["E_t"]
        er = data["E_R"]

        fig, ax1 = plt.subplots(figsize=(6, 4), dpi=150)
        ax2 = ax1.twinx()

        p1 = ax1.plot(poses, et, "b-o", label="Translation error $E_t$ [cm]")
        p2 = ax2.plot(poses, er, "r-s", label="Rotation error $E_R$ [deg]")

        ax1.set_xlabel("Number of Poses")
        ax1.set_ylabel("Translation Error [cm]", color="b")
        ax2.set_ylabel("Rotation Error [deg]", color="r")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.set_title("Fig. 6: Calibration Error vs. Number of Poses")

        lines = p1 + p2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right")

        plt.tight_layout()
        plt.savefig(save_p / "fig6_number_of_poses.png", bbox_inches="tight")
        plt.close()

    # Fig 7: Number of Image-Point Cloud Pairs
    if "pairs_study" in results_dict:
        data = results_dict["pairs_study"]
        pairs = data["x"]
        et = data["E_t"]
        er = data["E_R"]

        fig, ax1 = plt.subplots(figsize=(6, 4), dpi=150)
        ax2 = ax1.twinx()

        p1 = ax1.plot(pairs, et, "b-o", label="Translation error $E_t$ [cm]")
        p2 = ax2.plot(pairs, er, "r-s", label="Rotation error $E_R$ [deg]")

        ax1.set_xlabel("Number of Image-PCL Pairs")
        ax1.set_ylabel("Translation Error [cm]", color="b")
        ax2.set_ylabel("Rotation Error [deg]", color="r")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.set_title("Fig. 7: Calibration Error vs. Number of Image-PCL Pairs")

        lines = p1 + p2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right")

        plt.tight_layout()
        plt.savefig(save_p / "fig7_number_of_pairs.png", bbox_inches="tight")
        plt.close()
