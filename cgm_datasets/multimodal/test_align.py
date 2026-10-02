"""Evaluation: the reconstructed grid must match the prepared CGM trace, and modality resampling must produce grid-length aligned arrays."""

from utils.paths import DATA_ROOT, REPO_ROOT, CACHE_ROOT
import pickle

import numpy as np

from cgm_datasets.multimodal.align import (
    load_dexcom,
    reconstruct_grid,
    load_point_signal,
    resample_to_grid,
    resample_sleep,
)

D = str(DATA_ROOT)
DEX = D + "/wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6"
GAR = D + "/wearable_activity_monitor"


def _prepared(pid):
    recs = pickle.load(open(D + "/aireadi_cgm_full/aireadi_cgm_test.pkl", "rb"))
    return next(r for r in recs if str(r["person_id"]) == str(pid))


def test_grid_matches_prepared():
    pid = "1030"
    times, vals = load_dexcom(f"{DEX}/{pid}/{pid}_DEX.json")
    grid_t, grid_v, grid_m = reconstruct_grid(times, vals)
    rec = _prepared(pid)
    assert abs(len(grid_t) - rec["seq_len"]) <= 1, (len(grid_t), rec["seq_len"])
    assert int(grid_m.sum()) == len(vals), (int(grid_m.sum()), len(vals))


def test_grid_values_match_prepared():
    # Reconstructed glucose at observed slots should match prepared irg_ts (de-normed).
    pid = "1030"
    times, vals = load_dexcom(f"{DEX}/{pid}/{pid}_DEX.json")
    _, grid_v, grid_m = reconstruct_grid(times, vals)
    rec = _prepared(pid)
    n = min(len(grid_v), rec["seq_len"])
    prepared_mgdl = rec["irg_ts"][:n, 0] * 42.33 + 132.05
    obs = (grid_m[:n] > 0) & (rec["irg_ts_mask"][:n, 0] > 0)
    # allow tiny rounding from the prep's own normalization
    diff = np.abs(grid_v[:n][obs] - prepared_mgdl[obs])
    assert np.median(diff) < 1.0, float(np.median(diff))


def test_resample_hr_len():
    pid = "1030"
    times, vals = load_dexcom(f"{DEX}/{pid}/{pid}_DEX.json")
    grid_t, _, _ = reconstruct_grid(times, vals)
    pt, pv = load_point_signal(
        f"{GAR}/heart_rate/garmin_vivosmart5/{pid}/{pid}_heartrate.json",
        body_key="heart_rate",
        val_key="heart_rate",
        zero_is_missing=True,
    )
    v, m = resample_to_grid(grid_t, pt, pv, agg="mean")
    assert len(v) == len(grid_t) and len(m) == len(grid_t)
    assert m.min() >= 0 and m.max() <= 1
    assert m.sum() > 0.5 * len(grid_t)  # HR present for most of the grid


def test_sleep_onehot():
    pid = "1030"
    times, vals = load_dexcom(f"{DEX}/{pid}/{pid}_DEX.json")
    grid_t, _, _ = reconstruct_grid(times, vals)
    oh, pres = resample_sleep(
        f"{GAR}/sleep/garmin_vivosmart5/{pid}/{pid}_sleep.json", grid_t
    )
    assert oh.shape == (len(grid_t), 4)
    rs = oh.sum(1)
    assert set(np.unique(rs)).issubset({0.0, 1.0})
    assert pres.sum() > 0
