#!/usr/bin/env python3
"""Incrementally add the environment sensor (light, temperature) to an existing mm split, in place."""

from __future__ import annotations

from utils.paths import DATA_ROOT, REPO_ROOT, CACHE_ROOT

import json
import os
import pickle
import time

import numpy as np

from cgm_datasets.multimodal.align import (
    load_dexcom,
    reconstruct_grid,
    load_environment,
    resample_to_grid,
)

D = str(DATA_ROOT)
MM = D + "/aireadi_cgm_mm"
DEX = D + "/wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6"
ENV = D + "/environment/environmental_sensor/leelab_anura"


def _raw_env(pid, seq_len):
    """(light (T,), temp (T,), present (T,)) on the subject's 5-min grid; zeros if missing."""
    dexp = f"{DEX}/{pid}/{pid}_DEX.json"
    envp = f"{ENV}/{pid}/{pid}_ENV.csv"
    if not (os.path.exists(dexp) and os.path.exists(envp)):
        z = np.zeros(seq_len, np.float32)
        return z, z.copy(), z.copy()
    times, vals = load_dexcom(dexp)
    grid_t, _, _ = reconstruct_grid(times, vals)
    et, light, temp = load_environment(envp)
    lv, lp = resample_to_grid(grid_t, et, light, agg="mean")
    tv, tp = resample_to_grid(grid_t, et, temp, agg="mean")
    pres = ((lp > 0) | (tp > 0)).astype(np.float32)

    def fit(a):
        if len(a) == seq_len:
            return a
        return a[:seq_len] if len(a) > seq_len else np.pad(a, (0, seq_len - len(a)))

    return fit(lv), fit(tv), fit(pres)


def main():
    splits = ["train", "val", "test"]
    cache = {}
    acc = {"light": [0.0, 0.0, 0.0], "temp": [0.0, 0.0, 0.0]}  # sum, sumsq, cnt
    for sp in splits:
        recs = pickle.load(open(f"{MM}/aireadi_cgm_{sp}.pkl", "rb"))
        cache[sp] = recs
        t0 = time.time()
        for i, r in enumerate(recs):
            pid = str(r["person_id"])
            seq_len = int(r["seq_len"])
            light, temp, pres = _raw_env(pid, seq_len)
            r["_env_raw"] = (light, temp, pres)
            if sp == "train":
                m = pres > 0
                for name, arr in (("light", light), ("temp", temp)):
                    vv = arr[m]
                    acc[name][0] += float(vv.sum())
                    acc[name][1] += float((vv**2).sum())
                    acc[name][2] += float(m.sum())
            if (i + 1) % 200 == 0:
                print(f"  {sp}: {i+1}/{len(recs)} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[{sp}] env built ({time.time()-t0:.0f}s)", flush=True)

    st = {}
    for name in ("light", "temp"):
        c = max(acc[name][2], 1.0)
        mean = acc[name][0] / c
        var = max(acc[name][1] / c - mean**2, 1e-6)
        st[name] = {"mean": mean, "std": float(np.sqrt(var))}
    print("[env stats]", json.dumps(st), flush=True)

    for sp in splits:
        for r in cache[sp]:
            light, temp, pres = r.pop("_env_raw")
            ln = ((light - st["light"]["mean"]) / st["light"]["std"]).astype(np.float32)
            tn = ((temp - st["temp"]["mean"]) / st["temp"]["std"]).astype(np.float32)
            ln[pres <= 0] = 0.0
            tn[pres <= 0] = 0.0
            r["mm_env"] = np.stack([ln, tn], axis=1).astype(np.float32)  # (T,2)
            r["mm_env_p"] = pres.reshape(-1, 1).astype(np.float32)
        # atomic write: tmp + os.replace, so concurrent finetune readers never see a
        # partial file (rungs 0-4 ignore env, so old-or-new content is both fine).
        outp = f"{MM}/aireadi_cgm_{sp}.pkl"
        tmp = outp + ".tmp"
        pickle.dump(cache[sp], open(tmp, "wb"))
        os.replace(tmp, outp)
        print(f"[{sp}] env written -> {outp}", flush=True)

    nsp = f"{MM}/mm_norm_stats.json"
    cur = json.load(open(nsp)) if os.path.exists(nsp) else {}
    cur.setdefault("mm_stats", {}).update(
        {"env_light": st["light"], "env_temp": st["temp"]}
    )
    json.dump(cur, open(nsp, "w"), indent=2)
    print(f"updated {nsp}", flush=True)


if __name__ == "__main__":
    main()
