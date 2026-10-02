"""Simple imputation baselines: forward-fill and linear interpolation."""

from __future__ import annotations

import numpy as np


def forward_fill(series: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    """Forward fill then backward fill missing positions."""
    # Ensure mask is boolean and flatten if needed
    if obs_mask.ndim > 1:
        obs_mask = obs_mask[:, 0]
    obs_mask_bool = obs_mask.astype(bool)

    out = series.copy().astype(np.float32)
    if series.ndim > 1:
        out = out[:, 0]

    # Forward fill
    last_obs = None
    for i in range(len(out)):
        if obs_mask_bool[i]:
            last_obs = out[i]
        elif last_obs is not None:
            out[i] = last_obs
    # Backward fill for leading gaps
    last_obs = None
    for i in range(len(out) - 1, -1, -1):
        if obs_mask_bool[i]:
            last_obs = out[i]
        elif last_obs is not None and not obs_mask_bool[i]:
            out[i] = last_obs

    return out.reshape(-1, 1)


def linear_interp(series: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    """Linear (piecewise-linear) interpolation using np.interp.

    Classical numerical-analysis baseline with no single originating paper;
    see e.g. Davis, P. J. (1975). "Interpolation and Approximation." Dover.
    """
    # Handle input shapes
    if series.ndim > 1:
        series = series[:, 0]
    if obs_mask.ndim > 1:
        obs_mask = obs_mask[:, 0]

    idx = np.arange(len(series))
    obs_mask_bool = obs_mask.astype(bool)

    if obs_mask_bool.any():
        result = np.interp(idx, idx[obs_mask_bool], series[obs_mask_bool]).astype(
            np.float32
        )
    else:
        result = np.zeros(len(series), dtype=np.float32)

    return result.reshape(-1, 1)
