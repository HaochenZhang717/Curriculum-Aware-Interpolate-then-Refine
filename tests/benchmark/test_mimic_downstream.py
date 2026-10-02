# tests/benchmark/test_mimic_downstream.py
import importlib.util
import os

import numpy as np
from experiments.evaluation.mimic_downstream import make_battery


def _load_reference_mage():
    from utils.cgm_metrics import mage

    return mage


def test_mage_matches_reference():
    # MIMIC mage must equal the reference glucose mage on a fixed non-trivial trace,
    # so the two batteries stay cross-domain comparable.
    trace = np.array([80, 95, 70, 110, 65, 100, 72, 90], float)
    ref = _load_reference_mage()
    bat = make_battery(normal_lo=70.0, normal_hi=100.0, low_thr=65.0, high_thr=100.0)
    got = bat(trace)["mage"]
    assert abs(got - ref(trace)) < 1e-6


def test_battery_keys_match_downstream_summary():
    bat = make_battery(normal_lo=70.0, normal_hi=100.0, low_thr=65.0, high_thr=100.0)
    out = bat(np.array([60.0, 80.0, 90.0, 110.0, 70.0]))
    # keys must be a superset of SHAPE_MRR_KEYS + ABSERR_KEYS
    for k in ("tir", "tar180", "tbr70", "mage", "cv", "gmi", "mean"):
        assert k in out


def test_burden_thresholds_are_applied():
    bat = make_battery(normal_lo=70.0, normal_hi=100.0, low_thr=65.0, high_thr=100.0)
    g = np.array([60.0, 60.0, 80.0, 80.0])  # 50% below 65, 50% in-band
    out = bat(g)
    assert abs(out["tbr70"] - 50.0) < 1e-6  # percent below low threshold
    assert abs(out["tir"] - 50.0) < 1e-6  # percent within [70,100]
    assert abs(out["mean"] - 70.0) < 1e-6


def test_downstream_summary_consumes_battery():
    # end-to-end: the injected battery flows through downstream_summary
    from experiments.evaluation.toye_downstream import downstream_summary

    bat = make_battery(normal_lo=70.0, normal_hi=100.0, low_thr=65.0, high_thr=100.0)
    true = np.array([80.0, 62.0, 95.0, 105.0, 70.0, 68.0])
    method = true + 1.0
    meanfill = np.full_like(true, true.mean())
    d = downstream_summary(
        true_mgdl=true, method_mgdl=method, meanfill_mgdl=meanfill, battery=bat
    )
    for k in (
        "mrr_tir",
        "mrr_tar180",
        "mrr_tbr70",
        "mrr_mage",
        "mrr_cv",
        "gmi_abserr",
        "mean_abserr",
    ):
        assert k in d
