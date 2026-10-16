"""
Integration test for full MDPCalibPipeline and results persistence.
"""

from pathlib import Path
import numpy as np
import pytest
from mdpcalib.config import MDPCalibConfig
from mdpcalib.pipeline import MDPCalibPipeline
from mdpcalib.data.synthetic_data import generate_synthetic_calibration_data


def test_full_pipeline_execution(tmp_path):
    config = MDPCalibConfig()
    config.io.results_dir = str(tmp_path)
    config.io.run_name = "test_run"
    config.optimization.starting_pose = 5
    config.optimization.max_coarse_iterations = 40
    config.optimization.max_joint_iterations = 40

    data = generate_synthetic_calibration_data(
        num_poses=80,
        num_pairs=20,
        points_per_pair=200,
        camera="right",
        seed=42,
    )

    pipeline = MDPCalibPipeline(config=config)
    res = pipeline.run(
        T_cam_list=data["T_cam_list"],
        T_lidar_list=data["T_lidar_list"],
        pts_3d_lidar=data["pts_3d_lidar"],
        pts_2d_cmr=data["pts_2d_cmr"],
        K=data["K"],
        T_gt=data["T_gt"],
        output_dir=tmp_path,
    )

    out_dir = tmp_path / "test_run"
    assert (out_dir / "T_calib_init.txt").exists()
    assert (out_dir / "T_calib_final.txt").exists()
    assert (out_dir / "calibration_results.json").exists()
    assert (out_dir / "convergence_curve.png").exists()
    assert (out_dir / "point_cloud_projection.png").exists()

    metrics = res["final_metrics"]
    assert metrics.E_t_cm < 20.0, f"Final translation error {metrics.E_t_cm} should be < 20 cm for 80 poses"
    assert metrics.E_R_deg < 0.5, f"Final rotation error {metrics.E_R_deg} should be < 0.5 deg"
