"""
Configuration module for MDPCalib.
Provides structured configuration dataclasses and YAML loading/saving.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Dict, Any
import yaml


@dataclass
class OptimizationConfig:
    number_poses: int = 1000
    starting_pose: int = 15
    number_image_pcl_pairs: int = 100
    cauchy_param: float = 1e-3  # c parameter for Cauchy loss (c^2 = 1e-6 in Ceres)
    omega_rot_weight: float = 1.0  # Weight for rotation error matrix
    omega_trans_weight: float = 1.0  # Weight for translation error matrix
    omega_corr_weight: float = 1.0  # Weight for point correspondence error matrix
    initial_scale: float = 10.0  # Initial monocular VO scale factor estimate
    estimate_scale_per_pair: bool = False  # Global scale or per-pair scale
    max_coarse_iterations: int = 200
    max_joint_iterations: int = 200
    tolerance: float = 1e-8


@dataclass
class CMRNextConfig:
    amount_correspondences: float = 5.0  # Percentage of correspondences per pair [0, 100]
    rotation_threshold: float = 50.0  # [Deg] Max rotation difference threshold
    weights_paths: list = field(default_factory=lambda: [
        "checkpoints/cmrnext-calib-LEnc-iter1.tar",
        "checkpoints/cmrnext-calib-LEnc-iter5.tar",
        "checkpoints/cmrnext-calib-LEnc-iter6.tar",
    ])
    device: str = "cpu"  # "cuda" or "cpu"
    min_depth: float = 0.5  # meters
    max_depth: float = 80.0  # meters


@dataclass
class IOConfig:
    run_name: str = "kitti_seq00_right"
    results_dir: str = "results"
    save_plots: bool = True
    save_matrices: bool = True
    save_json: bool = True


@dataclass
class MDPCalibConfig:
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    cmrnext: CMRNextConfig = field(default_factory=CMRNextConfig)
    io: IOConfig = field(default_factory=IOConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "MDPCalibConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        opt_data = data.get("optimization", {})
        cmr_data = data.get("cmrnext", {})
        io_data = data.get("io", {})

        return cls(
            optimization=OptimizationConfig(**{k: v for k, v in opt_data.items() if k in OptimizationConfig.__annotations__}),
            cmrnext=CMRNextConfig(**{k: v for k, v in cmr_data.items() if k in CMRNextConfig.__annotations__}),
            io=IOConfig(**{k: v for k, v in io_data.items() if k in IOConfig.__annotations__}),
        )

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        p = Path(yaml_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "optimization": self.optimization.__dict__,
            "cmrnext": self.cmrnext.__dict__,
            "io": self.io.__dict__,
        }
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
