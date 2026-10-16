"""
MDPCalib: Automatic Target-Less Camera-LiDAR Calibration
From Motion and Deep Point Correspondences.
IEEE Robotics and Automation Letters (RA-L), Vol. 9, Issue 11, Nov 2024.
"""

from mdpcalib.pipeline import MDPCalibPipeline
from mdpcalib.config import MDPCalibConfig

__version__ = "1.0.0"
__all__ = ["MDPCalibPipeline", "MDPCalibConfig"]
