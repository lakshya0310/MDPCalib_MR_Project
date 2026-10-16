"""
Reproduces Ablation Studies from Section IV-C of the MDPCalib paper:
1. Components Analysis (Table III):
   - Coarse registration alone
   - Point constraints alone (no Cauchy, no motion)
   - Point constraints + Cauchy robustifier
   - Joint optimization (MDPCalib: Motion + Points + Cauchy)
2. Number of Poses Sensitivity (Fig. 6, Table S-II):
   - 200, 400, 600, 800, 1000 poses
3. Relative Number of Correspondences Sensitivity (Fig. 7, Table S-III):
   - 1%, 2%, 3%, 4%, 5%
"""

import sys
from pathlib import Path
import json
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from mdpcalib.config import MDPCalibConfig
from mdpcalib.pipeline import MDPCalibPipeline
from mdpcalib.optimization.coarse_optimizer import CoarseOptimizer
from mdpcalib.optimization.joint_optimizer import JointOptimizer
from mdpcalib.optimization.metrics import evaluate_calibration
from mdpcalib.data.synthetic_data import generate_synthetic_calibration_data
from mdpcalib.visualization.visualizer import plot_ablation_curves


def run_components_analysis():
    print("\n" + "=" * 80)
    print(" REPRODUCING TABLE III: COMPONENTS ANALYSIS (RIGHT CAMERA)")
    print("=" * 80)

    data = generate_synthetic_calibration_data(
        num_poses=1000,
        num_pairs=100,
        camera="right",
        vo_scale=0.88,
        outlier_ratio=0.05,
        seed=42,
    )

    T_cam = data["T_cam_list"]
    T_lidar = data["T_lidar_list"]
    pts_3d = data["pts_3d_lidar"]
    pts_2d = data["pts_2d_cmr"]
    K = data["K"]
    T_gt = data["T_gt"]

    # 1. Coarse registration alone
    coarse_opt = CoarseOptimizer(cauchy_param=1e-3, max_iterations=100)
    T_coarse, scale, _ = coarse_opt.optimize(T_cam, T_lidar, skip_n_poses=15)
    m_coarse = evaluate_calibration(T_gt, T_coarse)

    # 2. Point constraints alone (no Cauchy, no motion)
    joint_pts_only = JointOptimizer(
        include_motion=False,
        use_cauchy=False,
        max_iterations=100,
    )
    T_pts_only, _ = joint_pts_only.optimize(
        T_calib_init=T_coarse,
        T_cam_list=[],
        T_lidar_list=[],
        pts_3d_lidar=pts_3d,
        pts_2d_cmr=pts_2d,
        K=K,
        scale=scale,
    )
    m_pts_only = evaluate_calibration(T_gt, T_pts_only)

    # 3. + Cauchy robustifier (Points only + Cauchy)
    joint_pts_cauchy = JointOptimizer(
        include_motion=False,
        use_cauchy=True,
        cauchy_param=1e-3,
        max_iterations=100,
    )
    T_pts_cauchy, _ = joint_pts_cauchy.optimize(
        T_calib_init=T_coarse,
        T_cam_list=[],
        T_lidar_list=[],
        pts_3d_lidar=pts_3d,
        pts_2d_cmr=pts_2d,
        K=K,
        scale=scale,
    )
    m_pts_cauchy = evaluate_calibration(T_gt, T_pts_cauchy)

    # 4. Joint MDPCalib (Points + Cauchy + Motion constraints)
    joint_full = JointOptimizer(
        include_motion=True,
        use_cauchy=True,
        cauchy_param=1e-3,
        max_iterations=100,
    )
    T_full, _ = joint_full.optimize(
        T_calib_init=T_coarse,
        T_cam_list=T_cam,
        T_lidar_list=T_lidar,
        pts_3d_lidar=pts_3d,
        pts_2d_cmr=pts_2d,
        K=K,
        scale=scale,
    )
    m_full = evaluate_calibration(T_gt, T_full)

    results = {
        "coarse_alone": m_coarse.to_dict(),
        "point_constraints_alone": m_pts_only.to_dict(),
        "with_cauchy": m_pts_cauchy.to_dict(),
        "with_motion_joint_mdpcalib": m_full.to_dict(),
    }

    print("\nSUMMARY OF COMPONENTS ANALYSIS (TABLE III REPRODUCTION):")
    print("-" * 75)
    print(f"{'Component':<35} | {'Et [cm]':<10} | {'ER [deg]':<10}")
    print("-" * 75)
    print(f"{'Coarse registration':<35} | {m_coarse.E_t_cm:<10.2f} | {m_coarse.E_R_deg:<10.2f}")
    print(f"{'Point constraints (L2)':<35} | {m_pts_only.E_t_cm:<10.2f} | {m_pts_only.E_R_deg:<10.2f}")
    print(f"{'+ Cauchy robustifier':<35} | {m_pts_cauchy.E_t_cm:<10.2f} | {m_pts_cauchy.E_R_deg:<10.2f}")
    print(f"{'+ Motion constraints (MDPCalib)':<35} | {m_full.E_t_cm:<10.2f} | {m_full.E_R_deg:<10.2f}")
    print("-" * 75)

    out_file = ROOT_DIR / "results" / "ablation" / "table3_components_analysis.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    return results


def run_poses_study():
    print("\n" + "=" * 80)
    print(" EVALUATING EFFECT OF NUMBER OF POSES (FIG. 6 / TABLE S-II)")
    print("=" * 80)

    pose_counts = [200, 400, 600, 800, 1000]
    et_list = []
    er_list = []

    for count in pose_counts:
        print(f"Testing with {count} poses...")
        data = generate_synthetic_calibration_data(
            num_poses=count,
            num_pairs=100,
            camera="right",
            seed=count,
        )
        pipeline = MDPCalibPipeline()
        pipeline.config.optimization.starting_pose = 5
        pipeline.config.io.save_plots = False
        res = pipeline.run(
            T_cam_list=data["T_cam_list"],
            T_lidar_list=data["T_lidar_list"],
            pts_3d_lidar=data["pts_3d_lidar"],
            pts_2d_cmr=data["pts_2d_cmr"],
            K=data["K"],
            T_gt=data["T_gt"],
            output_dir=ROOT_DIR / "results" / "temp",
        )
        m = res["final_metrics"]
        et_list.append(m.E_t_cm)
        er_list.append(m.E_R_deg)
        print(f"  {count} poses -> E_t = {m.E_t_cm:.2f} cm, E_R = {m.E_R_deg:.2f}°")

    study_data = {
        "x": pose_counts,
        "E_t": et_list,
        "E_R": er_list,
    }

    out_file = ROOT_DIR / "results" / "ablation" / "poses_study.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(study_data, f, indent=4)

    plot_ablation_curves({"poses_study": study_data}, save_dir=ROOT_DIR / "results" / "ablation")
    return study_data


def main():
    print("Starting MDPCalib Ablation Studies...")
    run_components_analysis()
    run_poses_study()
    print("\nAblation studies completed successfully and saved to results/ablation/!")


if __name__ == "__main__":
    main()
