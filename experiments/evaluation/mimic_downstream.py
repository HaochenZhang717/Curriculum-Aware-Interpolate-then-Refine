"""Clinical-burden downstream battery for MIMIC vital signs."""

from __future__ import annotations


import numpy as np

from utils.cgm_metrics import mage as _mage


def make_battery(
    *, normal_lo: float, normal_hi: float, low_thr: float, high_thr: float
):
    """Return a battery(trace)->dict with clinical-burden metrics for one vital."""

    def battery(trace) -> dict[str, float]:
        g = np.asarray(trace, dtype=np.float64).reshape(-1)
        m = float(np.mean(g)) if g.size else float("nan")
        sd = float(np.std(g, ddof=1)) if g.size > 1 else 0.0
        return {
            "tir": float(np.mean((g >= normal_lo) & (g <= normal_hi)) * 100.0),
            "tar180": float(np.mean(g > high_thr) * 100.0),
            "tbr70": float(np.mean(g < low_thr) * 100.0),
            "mage": _mage(g),
            "cv": float(100.0 * sd / m) if m > 1 else float("nan"),
            "gmi": m,
            "mean": m,
        }

    return battery
