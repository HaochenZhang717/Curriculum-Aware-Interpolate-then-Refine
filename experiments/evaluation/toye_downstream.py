#!/usr/bin/env python3
"""Downstream clinical-metric recovery (MRR) reusing the existing battery."""

from __future__ import annotations


import numpy as np

SHAPE_MRR_KEYS = ["tir", "tar180", "tbr70", "mage", "cv"]
ABSERR_KEYS = ["gmi", "mean"]


def load_battery():
    """Return the CGM clinical metric battery."""
    from utils.metrics2 import battery

    return battery


def mrr_from_traces(true_mgdl, method_mgdl, meanfill_mgdl, battery=None):
    """Compute per-metric MRR for one reconstructed trace vs mean-fill."""
    if battery is None:
        battery = load_battery()
    bt = battery(np.asarray(true_mgdl, dtype=np.float64))
    bm = battery(np.asarray(method_mgdl, dtype=np.float64))
    bf = battery(np.asarray(meanfill_mgdl, dtype=np.float64))
    out = {}
    for k in bt:
        denom = abs(bf[k] - bt[k])
        num = abs(bm[k] - bt[k])
        floor = 1e-6 * (abs(bt[k]) + 1.0)
        out[k] = float("nan") if denom <= floor else 1.0 - num / denom
    return out


def downstream_summary(true_mgdl, method_mgdl, meanfill_mgdl, battery=None):
    """Compute stable shape-metric MRR values and absolute mean/GMI errors."""
    if battery is None:
        battery = load_battery()
    bt = battery(np.asarray(true_mgdl, dtype=np.float64))
    bm = battery(np.asarray(method_mgdl, dtype=np.float64))
    bf = battery(np.asarray(meanfill_mgdl, dtype=np.float64))
    out = {}
    for k in SHAPE_MRR_KEYS:
        denom = abs(bf[k] - bt[k])
        num = abs(bm[k] - bt[k])
        floor = 1e-6 * (abs(bt[k]) + 1.0)
        out[f"mrr_{k}"] = float("nan") if denom <= floor else 1.0 - num / denom
    for k in ABSERR_KEYS:
        out[f"{k}_abserr"] = float(abs(bm[k] - bt[k]))
    return out
