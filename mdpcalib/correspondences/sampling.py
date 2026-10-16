"""
Subsampling and filtering for 2D-3D point correspondences.
Implements:
- 5% random subset sampling per image-point cloud pair (Section III-C, IV-C)
- Filtering of out-of-bounds image pixels and points behind camera (Z <= 0.1)
- Spatial grid decimation to reduce redundant nearby constraints
"""

from typing import Tuple, Optional
import numpy as np


def filter_valid_projections(
    pts_3d: np.ndarray,
    pts_2d: np.ndarray,
    img_width: int,
    img_height: int,
    min_depth: float = 0.5,
    max_depth: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter correspondences to those strictly within valid image bounds and depth range.
    
    Args:
        pts_3d: (N, 3) 3D points in camera frame
        pts_2d: (N, 2) projected or measured pixel coordinates (u, v)
        img_width: Image width (e.g. 1242 for KITTI)
        img_height: Image height (e.g. 375 for KITTI)
        min_depth: Minimum depth in meters (default 0.5m)
        max_depth: Maximum depth in meters (default 80.0m matching paper Fig. 10)
        
    Returns:
        valid_pts_3d: (M, 3)
        valid_pts_2d: (M, 2)
        valid_mask: (N,) bool array
    """
    z = pts_3d[:, 2]
    u = pts_2d[:, 0]
    v = pts_2d[:, 1]

    mask = (
        (z >= min_depth)
        & (z <= max_depth)
        & (u >= 0.0)
        & (u < img_width)
        & (v >= 0.0)
        & (v < img_height)
        & (~np.isnan(u))
        & (~np.isnan(v))
        & (~np.isinf(u))
        & (~np.isinf(v))
    )

    return pts_3d[mask], pts_2d[mask], mask


def sample_correspondences(
    pts_3d: np.ndarray,
    pts_2d: np.ndarray,
    percentage: float = 5.0,
    seed: Optional[int] = None,
    max_count: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly subsample a percentage of correspondences per image-point cloud pair.
    Paper default: 5% of correspondences (~1,000 constraints per pair).
    
    Args:
        pts_3d: (N, 3) 3D points
        pts_2d: (N, 2) 2D pixel coordinates
        percentage: Value in [0.1, 100.0]
        seed: Random seed for deterministic reproducibility
        max_count: Optional hard upper bound on count
        
    Returns:
        sampled_3d: (K, 3)
        sampled_2d: (K, 2)
    """
    N = len(pts_3d)
    if N == 0:
        return pts_3d, pts_2d

    ratio = np.clip(percentage / 100.0, 0.001, 1.0)
    k = int(np.round(N * ratio))
    if max_count is not None:
        k = min(k, max_count)
    k = max(1, min(k, N))

    rng = np.random.default_rng(seed)
    indices = rng.choice(N, size=k, replace=False)

    return pts_3d[indices], pts_2d[indices]
