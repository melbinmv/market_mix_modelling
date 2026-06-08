"""
transforms.py
-------------
Reusable adstock and saturation transforms for the MMM model.
Imported by both model.py and pipeline.py.
"""

import numpy as np


def adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """
    Geometric adstock (carry-over).
    decay: float in [0, 1)
      0.0 → no carry-over (immediate effect only)
      0.8 → strong carry-over (TV-like)
    """
    if not (0.0 <= decay < 1.0):
        raise ValueError("decay must be in [0, 1)")
    out = np.zeros_like(x, dtype=float)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = x[t] + decay * out[t - 1]
    return out


def hill_saturation(x: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """
    Hill (diminishing returns) saturation.
    alpha : steepness — higher = more S-shaped curve
    gamma : half-saturation point — the spend level where effect = 50% of max
    Returns values in [0, 1].
    """
    x = np.array(x, dtype=float)
    return x**alpha / (x**alpha + gamma**alpha)


def apply_transforms(spend: np.ndarray, decay: float, alpha: float, gamma: float) -> np.ndarray:
    """Convenience: adstock → saturation in one call."""
    return hill_saturation(adstock(spend, decay), alpha, gamma)