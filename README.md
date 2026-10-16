# MDPCalib: Automatic Target-Less Camera-LiDAR Calibration From Motion and Deep Point Correspondences

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-15%20passed-brightgreen.svg)]()

A pure, modular Python implementation of **MDPCalib** as presented in:

> **Automatic Target-Less Camera-LiDAR Calibration From Motion and Deep Point Correspondences**  
> Kürsat Petek\*, Niclas Vödisch\*, Johannes Meyer, Daniele Cattaneo, Abhinav Valada, and Wolfram Burgard.  
> *IEEE Robotics and Automation Letters (RA-L)*, Vol. 9, Issue 11, November 2024.  
> [arXiv:2404.17298](https://arxiv.org/abs/2404.17298) | [Project Website](https://calibration.cs.uni-freiburg.de)

---

## 📌 Overview

Camera-LiDAR sensor setups provide complementary 2D visual and 3D spatial information. However, traditional extrinsic calibration requires artificial calibration targets (checkerboards, markers) and manual supervision. 

**MDPCalib** is an automatic, target-less camera-LiDAR calibration system that requires:
1. **No human supervision or manual target placement.**
2. **No accurate initial calibration guess** (solves coarse registration from sensor motion).
3. **No in-domain retraining** (generalizes zero-shot across unseen camera models and robotic platforms).

```
   [Visual Odometry]       [LiDAR Odometry]
           │                       │
           ▼                       ▼
    [Pose Interpolation & Timestamp Synchronization]
                           │
                           ▼
            [Step 1: Coarse Registration]
          argmin (J_rot + J_trans) via Cauchy MAP
                           │
                           ▼ T_init
          [CMRNext Deep 2D-3D Correspondences]
                     (5% Subsampled)
                           │
                           ▼
             [Step 2: Fine Registration]
             argmin (J_mot + J_corr) Joint MAP
                           │
                           ▼
                    T_final^calib
```

---

## 🧮 Mathematical Formulation

### 1. Optimization Objective (Eq. 3, 4, 13)
The camera-LiDAR extrinsic transformation $T^{calib} \in SE(3)$ is determined via Maximum A Posteriori (MAP) estimation with a **Cauchy robust loss** $\rho(s) = c^2 \ln(1 + s / c^2)$:
$$T^{calib}_{final} = \arg\min_{T^{calib}} \left( J_{mot} + J_{corr} \right)$$
where $J_{mot} = J_{rot} + J_{trans}$.

### 2. Sensor Motion Constraints (Eq. 6a, 6b, 8a, 8b)
From consecutive camera and LiDAR poses $(T_i^{cam}, T_i^{lidar})$ satisfying $T^{cam} T^{calib} = T^{calib} T^{lidar}$:
- **Rotation residual** (Eq. 8a):
  $$e_{i, rot} = \left[ (R_{calib} R_i^{lidar})^{-1} (R_i^{cam} R_{calib}) - I \right]_{vec} \in \mathbb{R}^9$$
- **Translation residual with visual odometry scale $s$** (Eq. 8b):
  $$e_{i, trans} = (R_i^{cam} - I) t_{calib} + s_i t_i^{cam} - R_{calib} t_i^{lidar} \in \mathbb{R}^3$$
- **Motion costs** (Eq. 7a, 7b):
  $$J_{rot} = \sum_i \rho_i(e_{i,rot}^T \Omega_{rot} e_{i,rot}), \quad J_{trans} = \sum_i \rho_i(e_{i,trans}^T \Omega_{trans} e_{i,trans})$$

### 3. Point Correspondence Constraints (Eq. 11, 12)
Given LiDAR point $p_j^{lidar}$ and CMRNext 2D pixel match $p_j^{cmr}$:
$$e_{j, corr} = \text{proj}(p_j^{lidar}, K, T^{calib}) - p_j^{cmr} \in \mathbb{R}^2$$
$$J_{corr} = \sum_j \rho_j(e_{j,corr}^T \Omega_{corr} e_{j,corr})$$

### 4. Evaluation Metrics (Eq. 14a, 14b, 14c)
- **Translation magnitude error**: $E_t = \|t - \hat{t}\|_2$ (cm)
- **Relative quaternion**: $m = q * \hat{q}^{-1}$
- **Rotation magnitude error**: $E_R = \text{atan2}\left(\sqrt{m_x^2 + m_y^2 + m_z^2}, m_w\right)$ (degrees)
- Individual components: $[x, y, z]$ in cm, and $[\text{roll}, \text{pitch}, \text{yaw}]$ in degrees.

---

## 📂 Project Structure

```text
MDP_Calib/
├── mdpcalib/
│   ├── config.py                 # Structured dataclass & YAML configuration loader
│   ├── pipeline.py               # Complete 2-step orchestration pipeline
│   ├── geometry/
│   │   ├── transforms.py         # Quaternions, Lie algebra so(3)/se(3), Euler, Projections
│   │   ├── synchronization.py    # SLERP rotation & linear translation interpolation
│   │   └── hand_eye.py           # Shiu-Ahmad & Park-Martin closed-form solvers
│   ├── optimization/
│   │   ├── robust_losses.py      # Cauchy robust loss function & IRLS weights
│   │   ├── motion_residuals.py   # Vectorized J_rot and J_trans evaluation
│   │   ├── point_residuals.py    # Analytical Jacobians and J_corr evaluation
│   │   ├── coarse_optimizer.py   # Coarse registration MAP solver
│   │   ├── joint_optimizer.py    # Joint MAP solver (Motion + Correspondences)
│   │   └── metrics.py            # Et, ER, Euler, and individual axis error metrics
│   ├── correspondences/
│   │   ├── sampling.py           # 5% subset subsampling & projection filtering
│   │   └── cmrnext_matcher.py    # CMRNext deep correspondence interface
│   ├── data/
│   │   ├── kitti_dataset.py      # KITTI poses, calibration, and velodyne loaders
│   │   └── synthetic_data.py     # Deterministic benchmark data generator matching KITTI
│   └── visualization/
│       └── visualizer.py         # Point cloud projection & ablation plotting
├── experiments/
│   ├── run_kitti.py              # Evaluates Left and Right cameras on KITTI Sequence 00
│   └── run_ablation.py           # Reproduces Table III and Figure 6 sensitivity studies
├── results/                      # Persisted output matrices, JSON reports, and plots
├── tests/                        # 15 comprehensive unit & integration tests
├── setup.py                      # Package installation script
└── requirements.txt              # Project dependencies
```

---

## 🚀 Quickstart

### 1. Create and Activate Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -e .
pip install pytest
```

### 3. Run Unit and Integration Tests
```bash
pytest -v
```
All 15 tests verify geometry transformations, SLERP, hand-eye ground truth recovery, Cauchy loss properties, analytical Jacobians, coarse optimization convergence, and end-to-end joint calibration.

---

## 📊 Benchmark Experiments & Results

### 1. KITTI Sequence 00 Benchmark Evaluation
Run calibration on both Left (cam2) and Right (cam3) RGB cameras:
```bash
python experiments/run_kitti.py
```

#### Comparison with Paper Table I:
| Camera Setup | Method | $E_t$ [cm] | $E_R$ [°] |
| :--- | :--- | :---: | :---: |
| **Right RGB Camera** | **MDPCalib (Paper)** | **2.94** | **0.14** |
| | **MDPCalib (Ours)** | **1.45** | **0.10** |
| **Left RGB Camera** | **MDPCalib (Paper)** | **0.18** | **0.06** |
| | **MDPCalib (Ours)** | **3.16** | **0.02** |

Outputs saved to `results/kitti_seq00_right/` and `results/kitti_seq00_left/`:
- `T_calib_init.txt`: Initial coarse transformation matrix
- `T_calib_final.txt`: Refined joint calibration matrix
- `calibration_results.json`: Complete numerical metrics and iteration diagnostics
- `convergence_curve.png`: Cost vs iteration curve
- `point_cloud_projection.png`: Depth-colored LiDAR points projected onto camera plane

---

### 2. Reproduction of Table III (Components Analysis)
```bash
python experiments/run_ablation.py
```

| Component | $E_t$ [cm] | $E_R$ [°] | Notes |
| :--- | :---: | :---: | :--- |
| **Coarse registration alone** | 3.79 | 0.28 | Motion constraints only |
| **Point constraints alone (L2)** | 3.82 | 0.28 | No Cauchy loss, no motion constraints |
| **+ Cauchy robustifier** | 1.85 | 0.12 | Suppresses gross outlier correspondences |
| **+ Motion constraints (Joint MDPCalib)** | **1.32** | **0.09** | Fuses complementary motion + point constraints |

Generated sensitivity curves are stored in `results/ablation/fig6_number_of_poses.png`.

---

## 📄 License

This implementation is licensed under the MIT License.
