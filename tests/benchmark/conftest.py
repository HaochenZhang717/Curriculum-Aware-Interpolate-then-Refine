import os, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def synthetic_day():
    """A deterministic 1-day (L=288) normalized CGM series + all-observed mask.

    A smooth diurnal sine so interpolation-style methods behave sanely, with a
    post-meal-like bump. Values are in z-space (mean 0, std ~1)."""
    L = 288
    t = np.arange(L)
    z = 0.6 * np.sin(2 * np.pi * t / 288) + 0.3 * np.sin(2 * np.pi * t / 48)
    series = z.astype(np.float32).reshape(L, 1)
    mask = np.ones((L, 1), dtype=np.float32)
    return series, mask
