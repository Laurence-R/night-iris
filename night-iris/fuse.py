"""Fusion: original everywhere except CLAHE on dark_mask.

dark_mask is already restricted to ~object_mask in sem, so objects stay original
without a second scatter.
"""
from __future__ import annotations

import numpy as np


def fuse(original: np.ndarray, clahe_img: np.ndarray, dark_mask: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 → same. One full-frame select; no fancy-index gather."""
    return np.where(dark_mask[..., np.newaxis], clahe_img, original)
