"""Recover the absolute UTC 5-min grid from raw Dexcom and resample modalities."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

STEP_MIN = 5
SLEEP_STAGES = ["awake", "light", "deep", "rem"]


def _parse(s: str) -> datetime:
    """ISO '2023-09-08T17:52:24Z' -> aware UTC datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def _to_float(x):
    """Parse a numeric value, returning NaN for '' / None / non-numeric."""
    try:
        if x is None or x == "":
            return np.nan
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _glucose(v):
    """Dexcom out-of-range markers: 'High' -> 400, 'Low' -> 40 mg/dL."""
    if isinstance(v, str):
        s = v.strip().lower()
        if s == "high":
            return 400.0
        if s == "low":
            return 40.0
        return float(v)
    return float(v)


def load_dexcom(path: str):
    """Return (sorted list of UTC datetimes, np.float32 glucose values mg/dL)."""
    cgm = json.load(open(path))["body"]["cgm"]
    times, vals = [], []
    for c in cgm:
        ti = c["effective_time_frame"]["time_interval"]
        times.append(_parse(ti["start_date_time"]))
        vals.append(_glucose(c["blood_glucose"]["value"]))
    order = np.argsort([t.timestamp() for t in times])
    times = [times[i] for i in order]
    vals = np.asarray(vals, np.float32)[order]
    return times, vals


def reconstruct_grid(times, vals, step_min: int = STEP_MIN):
    """Uniform 5-min grid spanning [first, last] Dexcom timestamp.

    Returns (grid_epoch_seconds (n,), grid_glucose (n,), grid_present (n,)).
    Missing 5-min slots have present=0.
    """
    t0 = times[0].timestamp()
    t1 = times[-1].timestamp()
    step = step_min * 60
    n = int(round((t1 - t0) / step)) + 1
    grid_t = t0 + np.arange(n) * step
    grid_v = np.zeros(n, np.float32)
    grid_m = np.zeros(n, np.float32)
    for t, v in zip(times, vals):
        i = int(round((t.timestamp() - t0) / step))
        if 0 <= i < n:
            grid_v[i] = v
            grid_m[i] = 1.0
    return grid_t.astype(np.float64), grid_v, grid_m


def load_point_signal(path, body_key, val_key, zero_is_missing=False):
    """Open-mHealth point signal: body[body_key][i][val_key]['value'] @ date_time.

    Returns (epoch_seconds (m,), values (m,) float32 with NaN where missing).
    """
    items = json.load(open(path))["body"][body_key]
    t, v = [], []
    for it in items:
        val = _to_float(it[val_key]["value"])
        if zero_is_missing and val == 0:
            val = np.nan
        t.append(_parse(it["effective_time_frame"]["date_time"]).timestamp())
        v.append(val)
    return np.asarray(t, np.float64), np.asarray(v, np.float32)


def load_interval_signal(path, body_key, val_fn):
    """Open-mHealth interval signal; timestamp = time_interval.start_date_time.

    `val_fn(item) -> float`. Returns (epoch_seconds (m,), values (m,) float32).
    """
    items = json.load(open(path))["body"][body_key]
    t, v = [], []
    for it in items:
        ti = it["effective_time_frame"]["time_interval"]
        t.append(_parse(ti["start_date_time"]).timestamp())
        v.append(_to_float(val_fn(it)))
    return np.asarray(t, np.float64), np.asarray(v, np.float32)


def resample_to_grid(grid_t, t, v, step_min: int = STEP_MIN, agg: str = "mean"):
    """Bin samples into the grid's 5-min slots.

    Returns (values (n,) float32, presence (n,) float32). presence=1 where the
    slot received >=1 finite sample. agg in {'mean','sum'}.
    """
    n = len(grid_t)
    out = np.zeros(n, np.float32)
    pres = np.zeros(n, np.float32)
    if len(t) == 0:
        return out, pres
    t0 = grid_t[0]
    step = step_min * 60
    idx = np.round((t - t0) / step).astype(np.int64)
    ok = (idx >= 0) & (idx < n) & np.isfinite(v)
    idx, vv = idx[ok], v[ok].astype(np.float64)
    sums = np.zeros(n)
    cnts = np.zeros(n)
    np.add.at(sums, idx, vv)
    np.add.at(cnts, idx, 1.0)
    nz = cnts > 0
    out[nz] = (sums[nz] if agg == "sum" else sums[nz] / cnts[nz]).astype(np.float32)
    pres[nz] = 1.0
    return out, pres


def load_environment(path):
    """leelab_anura env CSV (UTC `ts`, 5-sec). Returns (epoch_seconds, light, temp).

    light = mean of the visible-light channels (lch0..lch11); temp = ambient temp (C).
    """
    import pandas as pd

    lch = [f"lch{i}" for i in (0, 1, 2, 3, 6, 7, 8, 9, 10, 11)]
    df = pd.read_csv(
        path,
        comment="#",
        usecols=["ts", "temp"] + lch,
        na_values=["nan"],
        low_memory=False,
    )
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    ok = ts.notna().to_numpy()
    # version-robust epoch seconds: UTC -> naive ns -> int64
    naive = ts.dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype("int64")
    epoch = naive[ok] / 1e9
    light = df[lch].to_numpy(np.float32)[ok].mean(axis=1)
    temp = df["temp"].to_numpy(np.float32)[ok]
    return epoch.astype(np.float64), light.astype(np.float32), temp.astype(np.float32)


def resample_sleep(path, grid_t, step_min: int = STEP_MIN):
    """Sleep stage one-hot per slot. Returns (one_hot (n,4) float32, presence (n,))."""
    n = len(grid_t)
    oh = np.zeros((n, 4), np.float32)
    pres = np.zeros(n, np.float32)
    t0 = grid_t[0]
    step = step_min * 60
    for it in json.load(open(path))["body"]["sleep"]:
        ti = it["effective_time_frame"]["time_interval"]
        s = _parse(ti["start_date_time"]).timestamp()
        e = _parse(ti["end_date_time"]).timestamp()
        st = it.get("sleep_stage_state", "awake")
        c = SLEEP_STAGES.index(st) if st in SLEEP_STAGES else 0  # unknown -> awake
        i0 = max(0, int(round((s - t0) / step)))
        i1 = min(n, int(round((e - t0) / step)) + 1)
        if i1 > i0:
            oh[i0:i1] = 0.0
            oh[i0:i1, c] = 1.0
            pres[i0:i1] = 1.0
    return oh, pres
