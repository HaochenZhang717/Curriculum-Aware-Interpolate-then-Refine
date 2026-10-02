#!/usr/bin/env python3
"""Re-prepare the AI-READI CGM split with time-aligned wearable modality channels."""

from __future__ import annotations

from utils.paths import DATA_ROOT, REPO_ROOT, CACHE_ROOT

import argparse
import json
import os
import pickle
import time

import numpy as np

from cgm_datasets.multimodal.align import (
    load_dexcom,
    reconstruct_grid,
    load_point_signal,
    load_interval_signal,
    resample_to_grid,
    resample_sleep,
)

D = str(DATA_ROOT)
PREP = D + "/aireadi_cgm_full"
DEX = D + "/wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6"
GAR = D + "/wearable_activity_monitor"


# continuous point/interval signals: name -> (subdir, file_suffix, kind, body_key, val_key_or_fn, zero_missing, agg)
def _steps_val(it):  # interval signal value extractor
    return it["base_movement_quantity"]["value"]


SIGNALS = {
    "hr": (
        "heart_rate",
        "heartrate",
        "point",
        "heart_rate",
        "heart_rate",
        True,
        "mean",
    ),
    "steps": (
        "physical_activity",
        "activity",
        "interval",
        "activity",
        _steps_val,
        False,
        "sum",
    ),
    "cal": (
        "physical_activity_calorie",
        "calorie",
        "point",
        "activity",
        "calories_value",
        False,
        "sum",
    ),
    "resp": (
        "respiratory_rate",
        "respiratoryrate",
        "point",
        "breathing",
        "respiratory_rate",
        False,
        "mean",
    ),
    "stress": ("stress", "stress", "point", "stress", "stress", False, "mean"),
}
CONT = list(SIGNALS.keys())  # continuous modalities to z-norm


def _dex_path(pid):
    return f"{DEX}/{pid}/{pid}_DEX.json"


def _sig_path(pid, subdir, suffix):
    return f"{GAR}/{subdir}/garmin_vivosmart5/{pid}/{pid}_{suffix}.json"


def _fit_to_len(arr, n):
    """Trim or zero-pad an array (along axis 0) to exactly length n."""
    if len(arr) == n:
        return arr
    if len(arr) > n:
        return arr[:n]
    pad = [(0, n - len(arr))] + [(0, 0)] * (arr.ndim - 1)
    return np.pad(arr, pad)


def build_raw_modalities(pid, seq_len):
    """Return dict of raw (un-normalized) modality arrays for one subject, each
    length seq_len. Continuous: (n,) value + (n,) presence. sleep: (n,4)+presence.
    If raw Dexcom is missing, everything is zero with presence 0."""
    out = {}
    dexp = _dex_path(pid)
    if not os.path.exists(dexp):
        for k in CONT:
            out[k] = np.zeros(seq_len, np.float32)
            out[k + "_p"] = np.zeros(seq_len, np.float32)
        out["sleep"] = np.zeros((seq_len, 4), np.float32)
        out["sleep_p"] = np.zeros(seq_len, np.float32)
        out["env"] = np.zeros((seq_len, 2), np.float32)
        out["env_p"] = np.zeros(seq_len, np.float32)
        out["_has_grid"] = False
        return out
    times, vals = load_dexcom(dexp)
    grid_t, _, _ = reconstruct_grid(times, vals)
    for name, (subdir, suffix, kind, bkey, vkey, zmiss, agg) in SIGNALS.items():
        path = _sig_path(pid, subdir, suffix)
        if not os.path.exists(path):
            v = np.zeros(len(grid_t), np.float32)
            m = np.zeros(len(grid_t), np.float32)
        else:
            try:
                if kind == "point":
                    t, x = load_point_signal(path, bkey, vkey, zero_is_missing=zmiss)
                else:
                    t, x = load_interval_signal(path, bkey, vkey)
                v, m = resample_to_grid(grid_t, t, x, agg=agg)
            except Exception as e:
                print(f"  [warn] {pid} {name}: {e}", flush=True)
                v = np.zeros(len(grid_t), np.float32)
                m = np.zeros(len(grid_t), np.float32)
        out[name] = _fit_to_len(v, seq_len)
        out[name + "_p"] = _fit_to_len(m, seq_len)
    # sleep
    sp = _sig_path(pid, "sleep", "sleep")
    if os.path.exists(sp):
        try:
            oh, mp = resample_sleep(sp, grid_t)
        except Exception as e:
            print(f"  [warn] {pid} sleep: {e}", flush=True)
            oh = np.zeros((len(grid_t), 4), np.float32)
            mp = np.zeros(len(grid_t), np.float32)
    else:
        oh = np.zeros((len(grid_t), 4), np.float32)
        mp = np.zeros(len(grid_t), np.float32)
    out["sleep"] = _fit_to_len(oh, seq_len)
    out["sleep_p"] = _fit_to_len(mp, seq_len)

    out["env"] = np.zeros((seq_len, 2), np.float32)
    out["env_p"] = np.zeros(seq_len, np.float32)
    out["_has_grid"] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument(
        "--limit", type=int, default=0, help="process only N subjects/split (debug)"
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    # Pass A: load + build raw modalities; accumulate TRAIN continuous stats.
    cache = {}
    stats = {k: {"sum": 0.0, "sq": 0.0, "cnt": 0.0} for k in CONT}
    for sp in splits:
        recs = pickle.load(open(f"{PREP}/aireadi_cgm_{sp}.pkl", "rb"))
        if args.limit:
            recs = recs[: args.limit]
        cache[sp] = recs
        t0 = time.time()
        n_grid = 0
        for i, r in enumerate(recs):
            pid = str(r["person_id"])
            seq_len = int(r["seq_len"])
            raw = build_raw_modalities(pid, seq_len)
            r["_mm_raw"] = raw
            n_grid += int(raw["_has_grid"])
            # accumulate stats from train; if no train in this run, use all splits
            if sp == "train" or "train" not in splits:
                for k in CONT:
                    pres = raw[k + "_p"] > 0
                    vv = raw[k][pres]
                    stats[k]["sum"] += float(vv.sum())
                    stats[k]["sq"] += float((vv**2).sum())
                    stats[k]["cnt"] += float(pres.sum())
            if (i + 1) % 200 == 0:
                print(f"  {sp}: {i+1}/{len(recs)} ({time.time()-t0:.0f}s)", flush=True)
        print(
            f"[{sp}] built {len(recs)} records, {n_grid} with grid ({time.time()-t0:.0f}s)",
            flush=True,
        )

    # Compute train stats (fallback to current-split stats if train not in this run).
    mm_stats = {}
    for k in CONT:
        c = max(stats[k]["cnt"], 1.0)
        mean = stats[k]["sum"] / c
        var = max(stats[k]["sq"] / c - mean**2, 1e-6)
        mm_stats[k] = {"mean": mean, "std": float(np.sqrt(var))}
    if "train" not in splits:  # debug runs: self-normalize so values are sane
        print("[warn] no train split in this run; stats are from", splits)
    print("[stats]", json.dumps(mm_stats), flush=True)

    # Pass B: z-norm continuous, attach mm_* keys, drop raw, save.
    for sp in splits:
        recs = cache[sp]
        for r in recs:
            raw = r.pop("_mm_raw")
            for k in CONT:
                mean, std = mm_stats[k]["mean"], mm_stats[k]["std"]
                val = ((raw[k] - mean) / std).astype(np.float32)
                val[raw[k + "_p"] <= 0] = 0.0  # zero where absent
                r[f"mm_{k}"] = val.reshape(-1, 1)
                r[f"mm_{k}_p"] = raw[k + "_p"].reshape(-1, 1).astype(np.float32)
            r["mm_sleep"] = raw["sleep"].astype(np.float32)
            r["mm_sleep_p"] = raw["sleep_p"].reshape(-1, 1).astype(np.float32)
            r["mm_env"] = raw["env"].astype(np.float32)
            r["mm_env_p"] = raw["env_p"].reshape(-1, 1).astype(np.float32)
        outp = f"{args.out}/aireadi_cgm_{sp}.pkl"
        pickle.dump(recs, open(outp, "wb"))
        print(f"[{sp}] saved {outp} ({len(recs)} records)", flush=True)

    # Save mm stats alongside (merge into a norm_stats file).
    nsp = f"{args.out}/mm_norm_stats.json"
    json.dump({"mm_stats": mm_stats, "cont": CONT}, open(nsp, "w"), indent=2)
    print(f"saved {nsp}", flush=True)


if __name__ == "__main__":
    main()
