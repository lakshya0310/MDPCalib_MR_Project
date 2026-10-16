"""
Optimization subpackage for MDPCalib:
- Robust loss functions (Cauchy loss)
- Motion residuals (J_rot, J_trans)
- Point correspondence residuals (J_corr)
- Coarse and joint MAP optimizers
- Evaluation metrics
"""

from mdpcalib.optimization.robust_losses import cauchy_loss, cauchy_weight
from mdpcalib.optimization.motion_residuals import (
    compute_rotation_residual,
    compute_translation_residual,
    compute_motion_residuals,
)
from mdpcalib.optimization.point_residuals import (
    compute_point_residual,
    compute_point_residuals,
)
from mdpcalib.optimization.coarse_optimizer import CoarseOptimizer
from mdpcalib.optimization.joint_optimizer import JointOptimizer
from mdpcalib.optimization.metrics import CalibrationMetrics, evaluate_calibration

__all__ = [
    "cauchy_loss",
    "cauchy_weight",
    "compute_rotation_residual",
    "compute_translation_residual",
    "compute_motion_residuals",
    "compute_point_residual",
    "compute_point_residuals",
    "CoarseOptimizer",
    "JointOptimizer",
    "CalibrationMetrics",
    "evaluate_calibration",
]
