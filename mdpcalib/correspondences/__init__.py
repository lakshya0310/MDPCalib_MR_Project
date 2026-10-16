"""
Point correspondence extraction, projection, and subsampling subpackage.
"""

from mdpcalib.correspondences.sampling import (
    sample_correspondences,
    filter_valid_projections,
)
from mdpcalib.correspondences.cmrnext_matcher import CMRNextMatcher

__all__ = [
    "sample_correspondences",
    "filter_valid_projections",
    "CMRNextMatcher",
]
