"""
Robust loss functions for non-linear optimization.
Implements Cauchy robust loss (Liu & Tao [26], Ceres Solver):
rho(s) = c^2 * ln(1 + s / c^2)
with parameter c (c^2 = 1e-6 in MDPCalib paper/C++ code).
"""

from typing import Tuple, Union
import numpy as np


def cauchy_loss(s: Union[float, np.ndarray], c: float = 1e-3) -> Union[float, np.ndarray]:
    """
    Evaluate Cauchy loss rho(s) for squared residual norm s = ||e||^2 or e^T * Omega * e.
    rho(s) = c^2 * ln(1 + s / c^2)
    """
    c2 = c * c
    return c2 * np.log1p(np.maximum(0.0, s) / c2)


def cauchy_derivatives(s: Union[float, np.ndarray], c: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """
    First and second derivatives of Cauchy loss with respect to s:
    rho'(s) = 1 / (1 + s / c^2)
    rho''(s) = -1 / (c^2 * (1 + s / c^2)^2)
    """
    c2 = c * c
    denom = 1.0 + np.maximum(0.0, s) / c2
    rho_prime = 1.0 / denom
    rho_double_prime = -1.0 / (c2 * denom * denom)
    return rho_prime, rho_double_prime


def cauchy_weight(residual_norm: Union[float, np.ndarray], c: float = 1e-3) -> Union[float, np.ndarray]:
    """
    Equivalent weight w(r) for Iteratively Reweighted Least Squares (IRLS):
    w(r) = sqrt(rho'(r^2)) = 1 / sqrt(1 + (r/c)^2)
    So that weighted residual is r_eff = w(r) * r.
    """
    s = residual_norm ** 2
    c2 = c * c
    return 1.0 / np.sqrt(1.0 + s / c2)
