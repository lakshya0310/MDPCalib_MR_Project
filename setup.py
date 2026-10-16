from setuptools import setup, find_packages

setup(
    name="mdpcalib",
    version="1.0.0",
    description="Automatic Target-Less Camera-LiDAR Calibration From Motion and Deep Point Correspondences (IEEE RA-L 2024)",
    author="MDPCalib Contributors",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "deep": ["torch>=2.0.0", "torchvision>=0.15.0", "opencv-python>=4.7.0"],
        "dev": ["pytest>=7.0.0", "ruff>=0.1.0"],
    },
)
