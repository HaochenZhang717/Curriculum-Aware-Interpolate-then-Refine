#!/usr/bin/env python3
"""Toye et al. imputation metrics: RMSE, bias, EmpSE, in mg/dL."""

from __future__ import annotations

import numpy as np


def _err_mgdl(pred, true, target_mask, std_mgdl):
    m = np.asarray(target_mask).astype(bool)
    if not m.any():
        return None
    return (np.asarray(pred)[m] - np.asarray(true)[m]) * std_mgdl


def rmse_mgdl(pred, true, target_mask, std_mgdl):
    e = _err_mgdl(pred, true, target_mask, std_mgdl)
    if e is None:
        return float("nan")
    return float(np.sqrt((e**2).mean()))


def bias_mgdl(pred, true, target_mask, std_mgdl):
    e = _err_mgdl(pred, true, target_mask, std_mgdl)
    if e is None:
        return float("nan")
    return float(e.mean())


def empse_mgdl(pred, true, target_mask, std_mgdl):
    """Empirical SE = std (ddof=1) of the imputed predictions, in mg/dL."""
    m = np.asarray(target_mask).astype(bool)
    if m.sum() < 2:
        return float("nan")
    preds_mgdl = np.asarray(pred)[m] * std_mgdl
    return float(np.std(preds_mgdl, ddof=1))
