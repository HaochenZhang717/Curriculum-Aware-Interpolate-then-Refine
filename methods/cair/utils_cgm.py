#!/usr/bin/env python3
"""Lightweight CGM utility functions with NO neural network dependencies."""

from __future__ import annotations
import numpy as np

CGM_MEAN = 132.05
CGM_STD = 42.33
DAY_LEN = 288  # 5-min intervals per day


def compute_diurnal_curve_tod(
    train_samples: list[dict], day_len: int = DAY_LEN
) -> np.ndarray:
    """
    Compute population-mean time-of-day glucose curve (length day_len).
    Returns a (day_len,) array in normalized glucose units.
    """
    accum = np.zeros(day_len, dtype=np.float64)
    count = np.zeros(day_len, dtype=np.float64)
    for s in train_samples:
        ts = s["irg_ts"][:, 0].astype(np.float64)
        m = s["irg_ts_mask"][:, 0].astype(np.float64)
        L = len(ts)
        tod = np.arange(L) % day_len
        np.add.at(accum, tod, ts * m)
        np.add.at(count, tod, m)
    count[count == 0] = 1.0
    return (accum / count).astype(np.float32)


def denormalize(x: np.ndarray) -> np.ndarray:
    """Convert normalized CGM values back to mg/dL."""
    return x * CGM_STD + CGM_MEAN
