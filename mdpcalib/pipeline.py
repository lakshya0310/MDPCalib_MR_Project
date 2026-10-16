"""
End-to-End MDPCalib Pipeline.
Implements the full 2-step calibration architecture from the paper:
1. Coarse Registration (Sensor Motion Alignment via Visual & LiDAR Odometry)
2. Fine Registration (Joint Optimization with Deep Point Correspondences)
"""

from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
import json
import numpy as np

from mdpcalib.config import MDPCalibConfig
from mdpcalib.geometry.transforms import make_transform, invert_transform
from mdpcalib.geometry.synchronization import (
    synchronize_trajectories,
    compute_relative_motion_pairs,
)
from mdpcalib.optimization.coarse_optimizer import CoarseOptimizer
from mdpcalib.optimization.joint_optimizer import JointOptimizer
from mdpcalib.optimization.metrics import CalibrationMetrics, evaluate_calibration
from mdpcalib.correspondences.cmrnext_matcher import CMRNextMatcher
from mdpcalib.visualization.visualizer import (
    project_points_to_image,
    plot_convergence,
)


class MDPCalibPipeline:
    """
    Main orchestration class for MDPCalib calibration.
    """

    def __init__(self, config: Optional[MDPCalibConfig] = None):
        self.config = config or MDPCalibConfig()
        self.coarse_optimizer = CoarseOptimizer(
            cauchy_param=self.config.optimization.cauchy_param,
            omega_rot=self.config.optimization.omega_rot_weight,
            omega_trans=self.config.optimization.omega_trans_weight,
            max_iterations=self.config.optimization.max_coarse_iterations,
            tolerance=self.config.optimization.tolerance,
            initial_scale=self.config.optimization.initial_scale,
        )
        self.joint_optimizer = JointOptimizer(
            cauchy_param=self.config.optimization.cauchy_param,
            omega_rot=self.config.optimization.omega_rot_weight,
            omega_trans=self.config.optimization.omega_trans_weight,
            omega_corr=self.config.optimization.omega_corr_weight,
            max_iterations=self.config.optimization.max_joint_iterations,
            tolerance=self.config.optimization.tolerance,
        )
        self.cmr_matcher = CMRNextMatcher(
            weights_paths=self.config.cmrnext.weights_paths,
            device=self.config.cmrnext.device,
            amount_correspondences=self.config.cmrnext.amount_correspondences,
            min_depth=self.config.cmrnext.min_depth,
            max_depth=self.config.cmrnext.max_depth,
        )

    def run(
        self,
        T_cam_list: List[np.ndarray],
        T_lidar_list: List[np.ndarray],
        pts_3d_lidar: np.ndarray,
        K: np.ndarray,
        pts_2d_cmr: Optional[np.ndarray] = None,
        img: Optional[np.ndarray] = None,
        T_gt: Optional[np.ndarray] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, any]:
        """
        Execute full MDPCalib calibration pipeline.
        
        Args:
            T_cam_list: List of relative camera motion matrices (4x4)
            T_lidar_list: List of relative LiDAR motion matrices (4x4)
            pts_3d_lidar: (N, 3) 3D LiDAR point cloud
            K: (3, 3) Camera intrinsic matrix
            pts_2d_cmr: Optional precomputed 2D CMRNext points (if None, predicted by CMRNext)
            img: Optional RGB image
            T_gt: Optional 4x4 ground truth extrinsic calibration matrix
            output_dir: Output directory to store results
            
        Returns:
            dict containing calibration results, metrics, and diagnostics.
        """
        out_dir = Path(output_dir or self.config.io.results_dir) / self.config.io.run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("Starting MDPCalib Extrinsic Camera-LiDAR Calibration")
        print(f"Number of sensor motion poses: {len(T_cam_list)}")
        print(f"Number of 3D LiDAR points: {len(pts_3d_lidar)}")
        print("=" * 70)

        # -------------------------------------------------------------
        # STEP 1: Coarse Registration (Sensor Motion Alignment)
        # -------------------------------------------------------------
        print("\n[Step 1/2] Running Coarse Registration from Sensor Motion...")
        skip_n = self.config.optimization.starting_pose
        T_calib_init, final_scale, coarse_info = self.coarse_optimizer.optimize(
            T_cam_list=T_cam_list,
            T_lidar_list=T_lidar_list,
            skip_n_poses=skip_n,
        )

        print(f" Coarse optimization finished in {coarse_info['iterations']} iterations.")
        print(f" Estimated VO metric scale factor s: {final_scale:.4f}")
        print(" Coarse calibration matrix T_calib_init:")
        print(np.array2string(T_calib_init, precision=4, suppress_small=True))

        coarse_metrics = None
        if T_gt is not None:
            coarse_metrics = evaluate_calibration(T_gt, T_calib_init)
            print(" Coarse Calibration Error:")
            print(f"   {coarse_metrics.summary_str()}")

        # -------------------------------------------------------------
        # STEP 2: Fine Registration (Deep Point Correspondences)
        # -------------------------------------------------------------
        print("\n[Step 2/2] Running Fine Registration via Joint Optimization...")

        # If 2D correspondences not precomputed, predict using CMRNext
        if pts_2d_cmr is None:
            print(" Extracting 2D-to-3D deep point correspondences using CMRNext...")
            pts_3d_sub, pts_2d_sub = self.cmr_matcher.predict_correspondences(
                pts_3d_lidar=pts_3d_lidar,
                img=img,
                K=K,
                T_init=T_calib_init,
                subsample_percentage=self.config.cmrnext.amount_correspondences,
            )
        else:
            pts_3d_sub = pts_3d_lidar
            pts_2d_sub = pts_2d_cmr

        print(f" Using {len(pts_3d_sub)} correspondence constraints (5% sampled).")

        T_calib_final, joint_info = self.joint_optimizer.optimize(
            T_calib_init=T_calib_init,
            T_cam_list=T_cam_list,
            T_lidar_list=T_lidar_list,
            pts_3d_lidar=pts_3d_sub,
            pts_2d_cmr=pts_2d_sub,
            K=K,
            scale=final_scale,
        )

        print(f" Joint optimization finished in {joint_info['iterations']} iterations.")
        print(" Final calibration matrix T_calib_final:")
        print(np.array2string(T_calib_final, precision=4, suppress_small=True))

        final_metrics = None
        if T_gt is not None:
            final_metrics = evaluate_calibration(T_gt, T_calib_final)
            print(" Final Calibration Error (MDPCalib):")
            print(f"   {final_metrics.summary_str()}")

        # -------------------------------------------------------------
        # Store Results & Artifacts
        # -------------------------------------------------------------
        print(f"\n[Artifacts] Storing calibration results to: {out_dir}")

        # Save matrices
        np.savetxt(out_dir / "T_calib_init.txt", T_calib_init, fmt="%.8e")
        np.savetxt(out_dir / "T_calib_final.txt", T_calib_final, fmt="%.8e")

        # Save JSON summary
        summary = {
            "run_name": self.config.io.run_name,
            "scale": float(final_scale),
            "coarse_iterations": coarse_info["iterations"],
            "joint_iterations": joint_info["iterations"],
            "num_motion_poses": len(T_cam_list),
            "num_correspondences": len(pts_3d_sub),
            "T_calib_init": T_calib_init.tolist(),
            "T_calib_final": T_calib_final.tolist(),
        }
        if coarse_metrics is not None:
            summary["coarse_metrics"] = coarse_metrics.to_dict()
        if final_metrics is not None:
            summary["final_metrics"] = final_metrics.to_dict()

        with open(out_dir / "calibration_results.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        # Save plots
        if self.config.io.save_plots:
            plot_convergence(
                coarse_history=coarse_info["cost_history"],
                joint_history=joint_info["cost_history"],
                save_path=out_dir / "convergence_curve.png",
            )
            project_points_to_image(
                pts_3d=pts_3d_lidar,
                img=img,
                K=K,
                T_calib=T_calib_final,
                save_path=out_dir / "point_cloud_projection.png",
            )

        print("Calibration completed successfully!")
        return {
            "T_calib_init": T_calib_init,
            "T_calib_final": T_calib_final,
            "scale": final_scale,
            "coarse_metrics": coarse_metrics,
            "final_metrics": final_metrics,
            "coarse_info": coarse_info,
            "joint_info": joint_info,
            "output_dir": str(out_dir),
        }
