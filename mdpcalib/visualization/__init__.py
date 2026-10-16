"""
Visualization subpackage for MDPCalib:
- Depth-colored LiDAR projection onto RGB images
- Optimization convergence curves
- Ablation study plots (Figures 6, 7, 8, 9)
"""

from mdpcalib.visualization.visualizer import (
    project_points_to_image,
    plot_convergence,
    plot_ablation_curves,
)

__all__ = [
    "project_points_to_image",
    "plot_convergence",
    "plot_ablation_curves",
]
