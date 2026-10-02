"""Gap-sensitive CGM metric battery for the hard downstream eval."""

from __future__ import annotations
import os
import sys
import numpy as np

from utils import cgm_metrics as CM

NIGHT = slice(0, 72)  # 00:00-06:00
DAWN_NADIR = slice(0, 84)  # 00:00-07:00
DAWN_RISE = slice(72, 108)  # 06:00-09:00


def adrr(g):
    f = CM._risk_transform(np.asarray(g, float))
    rl = np.where(f < 0, 10.0 * f * f, 0.0)
    rh = np.where(f > 0, 10.0 * f * f, 0.0)
    return float(rl.max() + rh.max()) if len(f) else float("nan")


def conga1(g, k=12):
    g = np.asarray(g, float)
    if len(g) <= k:
        return float("nan")
    d = g[k:] - g[:-k]
    return float(np.std(d, ddof=1)) if len(d) > 1 else 0.0


def dawn(g):
    g = np.asarray(g, float)
    if len(g) < 108:
        return float("nan")
    return float(np.mean(g[DAWN_RISE]) - np.min(g[DAWN_NADIR]))


def noct_tbr54(g):
    g = np.asarray(g, float)
    return float(np.mean(g[NIGHT] < 54) * 100) if len(g) >= 72 else float("nan")


def noct_tbr70(g):
    g = np.asarray(g, float)
    return float(np.mean(g[NIGHT] < 70) * 100) if len(g) >= 72 else float("nan")


def noct_lbgi(g):
    g = np.asarray(g, float)
    if len(g) < 72:
        return float("nan")
    f = CM._risk_transform(g[NIGHT])
    rl = np.where(f < 0, 10.0 * f * f, 0.0)
    return float(np.mean(rl))


# added metrics keyed like CM, with group + pretty label
ADDED = [
    ("adrr", adrr, "a.u.", "risk", "ADRR"),
    ("conga1", conga1, "mg/dL", "variability", "CONGA-1h"),
    ("dawn", dawn, "mg/dL", "positional", "Dawn rise"),
    ("noct_tbr54", noct_tbr54, "%", "hypo", "Noct. TBR<54"),
    ("noct_tbr70", noct_tbr70, "%", "hypo", "Noct. TBR<70"),
    ("noct_lbgi", noct_lbgi, "a.u.", "hypo", "Noct. LBGI"),
]

# full battery = CM battery (17) + added (6) = 23 metrics
KEYS = CM.METRIC_KEYS + [k for k, *_ in ADDED]
GROUP = {**CM.METRIC_GROUP, **{k: g for k, _f, _u, g, _l in ADDED}}
LABEL = {**CM.METRIC_LABELS, **{k: l for k, _f, _u, _g, l in ADDED}}
UNITS = {**CM.METRIC_UNITS, **{k: u for k, _f, u, _g, _l in ADDED}}
# metrics whose signal is shape/tail/overnight (where smooth fills fail) vs robust "floor"
SHAPE_TAIL = [
    "sd",
    "cv",
    "mage",
    "jindex",
    "conga1",
    "lbgi",
    "hbgi",
    "tbr54",
    "tbr70",
    "tar250",
    "gri",
    "gri_hypo",
    "gri_hyper",
    "adrr",
    "auc180",
    "dawn",
    "noct_tbr54",
    "noct_tbr70",
    "noct_lbgi",
]
FLOOR = ["mean", "gmi", "tir", "tar180"]  # mean-driven, robust to short MCAR dropout


def battery(g_mgdl) -> dict:
    g = np.asarray(g_mgdl, dtype=np.float64)
    out = CM.metric_vector_mgdl(g)
    for k, fn, *_ in ADDED:
        out[k] = fn(g)
    return out
