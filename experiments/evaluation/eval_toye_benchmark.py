#!/usr/bin/env python3
"""Toye real-world-missingness benchmark orchestrator."""

from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATASET_ROOT = os.path.join(ROOT, "data")
DEFAULT_MM = os.path.join(DATASET_ROOT, "aireadi_cgm_mm", "aireadi_cgm_test.pkl")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.missingness_mechanisms import (
    create_mcar_mask,
    create_mar_mask,
    create_nmar_mask,
    native_gap_durations,
)
from baselines.toye.toye_baselines import (
    impute_linear,
    impute_locf,
    impute_mean,
    impute_mode,
    impute_knn,
    impute_hotdeck,
    impute_mice,
    impute_missforest,
    impute_fourier,
)
from experiments.evaluation.toye_metrics import rmse_mgdl, bias_mgdl, empse_mgdl
from experiments.evaluation.toye_downstream import downstream_summary, load_battery
from cgm_datasets import load_dataset_splits

PERCENTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
MECHS = ["mcar", "mar", "nmar"]

INPROC_BASELINES = {
    "linear": impute_linear,
    "locf": impute_locf,
    "mean": impute_mean,
    "mode": impute_mode,
    "knn": impute_knn,
    "hotdeck": impute_hotdeck,
    "mice": impute_mice,
    "missforest": impute_missforest,
    "fourier": impute_fourier,
}


def aggregate_cells(cells, value_keys):
    """Mean each value_key over records sharing (method, mech, pct)."""
    buckets = defaultdict(lambda: defaultdict(list))
    for c in cells:
        key = (c["method"], c["mech"], c["pct"])
        for vk in value_keys:
            if vk in c and c[vk] == c[vk]:  # skip NaN
                buckets[key][vk].append(c[vk])
    out = {}
    for key, d in buckets.items():
        out[key] = {vk: float(np.mean(v)) for vk, v in d.items()}
    return out


def load_mm_activity(mm_pkl):
    import pickle

    if not os.path.exists(mm_pkl):
        return {}
    recs = pickle.load(open(mm_pkl, "rb"))
    out = {}
    for r in recs:
        pid = int(r["person_id"])
        steps = np.asarray(r["mm_steps"]).reshape(-1).astype(np.float32)
        hr = np.asarray(r["mm_hr"]).reshape(-1).astype(np.float32)
        cal = np.asarray(r["mm_cal"]).reshape(-1).astype(np.float32)
        out[pid] = {"steps": steps, "hr": hr, "cal": cal}
    return out


def load_group_map():
    import os
    import pandas as pd

    p = os.path.join(DATASET_ROOT, "participants.tsv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p, sep="\t")
    return {int(r): str(g) for r, g in zip(df["person_id"], df["study_group"])}


def _make_masks(
    series, obs_mask, mech, pct, trigger, gap_durs, seed, mean_mgdl, std_mgdl
):
    if mech == "mcar":
        return create_mcar_mask(series, obs_mask, percent=pct, seed=seed)
    if mech == "mar":
        return create_mar_mask(
            series,
            obs_mask,
            trigger=trigger,
            percent=pct,
            gap_durations=gap_durs,
            seed=seed,
        )
    return create_nmar_mask(
        series,
        obs_mask,
        percent=pct,
        gap_durations=gap_durs,
        seed=seed,
        mean_mgdl=mean_mgdl,
        std_mgdl=std_mgdl,
    )


def _run_pypots(masked_records, pypots_python, method, tmpdir):
    """masked_records: list of dicts with 'X' (L,1) NaN-at-missing. Returns list
    of filled (L,1) arrays aligned to input order. Runs the isolated runner."""
    X = np.stack([r["X"] for r in masked_records], axis=0)  # (N,L,1)
    inp = os.path.join(tmpdir, f"{method}_in.npz")
    out = os.path.join(tmpdir, f"{method}_out.npz")
    np.savez(inp, X=X)
    runner = os.path.join(ROOT, "baselines", "toye", "pypots_runner.py")
    subprocess.run(
        [pypots_python, runner, "--in", inp, "--out", out, "--method", method],
        check=True,
    )
    return list(np.load(out)["X_imputed"])


def run(args):
    import torch

    device = (
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    _, _, test, meta = load_dataset_splits(args.dataset, prepared_dir=args.data_dir)
    test = test[: args.n_participants]
    std_mgdl = float(meta.glucose_std_mgdl)
    mean_mgdl = float(meta.glucose_mean_mgdl)
    t_lo, t_hi = meta.eval_window

    from methods.refine import RefineImputer
    import glob as _glob

    rpaths = sorted(_glob.glob(os.path.join(ROOT, args.refine_ckpt_glob)))[
        : args.n_refine_members
    ]
    assert rpaths, f"no REFINE checkpoints matched {args.refine_ckpt_glob}"
    refine = RefineImputer(ckpt_paths=rpaths, device=device)
    battery = load_battery()
    mm_activity = load_mm_activity(args.mm_pkl)
    group_map = load_group_map()

    cells = []
    pypots_jobs = []
    mar_skipped = 0
    for pi, samp in enumerate(test):
        pid = int(samp["person_id"])
        grp = group_map.get(pid, "unknown")
        full_ts = samp["irg_ts"][:, 0].astype(np.float32)
        full_orig = samp["irg_ts_mask"][:, 0].astype(bool)
        d2 = samp["irg_ts"][t_lo:t_hi].copy()
        m2 = samp["irg_ts_mask"][t_lo:t_hi].copy()
        true_d2 = d2[:, 0].astype(np.float32)
        true_mgdl = true_d2 * std_mgdl + mean_mgdl
        gap_durs = native_gap_durations(samp["irg_ts_mask"]) or [3, 6, 9, 12]
        trigger = _activity_trigger(samp, t_lo, t_hi, mm_activity)
        if trigger is None:
            mar_skipped += 1
        meanfill = np.full_like(true_mgdl, float(true_mgdl.mean()))

        for mech in MECHS:
            if mech == "mar" and trigger is None:
                continue  # logged once below
            for pct in PERCENTS:
                seed = 1000 * pi + 100 * MECHS.index(mech) + int(pct * 100)
                em, tm = _make_masks(
                    d2.copy(),
                    m2.copy(),
                    mech,
                    pct,
                    trigger,
                    gap_durs,
                    seed,
                    mean_mgdl,
                    std_mgdl,
                )
                obs = em[:, 0].astype(bool)
                tgt = tm[:, 0].astype(bool)
                if not tgt.any():
                    continue
                seen = np.where(obs, true_d2, np.nan)
                pypots_jobs.append(
                    {
                        "X": np.where(obs, true_d2, np.nan)
                        .reshape(-1, 1)
                        .astype(np.float32),
                        "mech": mech,
                        "pct": pct,
                        "pi": pi,
                        "pid": pid,
                        "group": grp,
                        "true": true_d2,
                        "tgt": tgt,
                        "meanfill": meanfill,
                    }
                )
                for name, fn in INPROC_BASELINES.items():
                    pred = fn(seen, obs.astype(np.float32))
                    cell = _score(
                        name,
                        mech,
                        pct,
                        pred,
                        true_d2,
                        tgt,
                        std_mgdl,
                        mean_mgdl,
                        meanfill,
                        battery,
                        group=grp,
                        pid=pid,
                    )
                    cells.append(cell)
                # REFINE (operates on the full series)
                fe = full_orig.copy()
                fe[t_lo:t_hi] = obs
                fm = full_ts.copy()
                fm[~fe] = 0.0
                rpred = refine.impute(fm, fe.astype(np.float32))[t_lo:t_hi]
                cells.append(
                    _score(
                        "refine",
                        mech,
                        pct,
                        rpred,
                        true_d2,
                        tgt,
                        std_mgdl,
                        mean_mgdl,
                        meanfill,
                        battery,
                        group=grp,
                        pid=pid,
                    )
                )
        if pi % 10 == 0:
            print(f"  pid {pi}/{len(test)}", flush=True)

    def _write(skipped):
        agg = aggregate_cells(
            cells,
            value_keys=[
                "rmse",
                "bias",
                "empse",
                "mrr_tir",
                "mrr_tar180",
                "mrr_tbr70",
                "mrr_mage",
                "mrr_cv",
                "gmi_abserr",
                "mean_abserr",
            ],
        )
        out = {
            "dataset": meta.name,
            "n_participants": len(test),
            "mar_skipped_persons": mar_skipped,
            "n_refine_members": len(rpaths),
            "cells": [
                {
                    k: c[k]
                    for k in (
                        "method",
                        "mech",
                        "pct",
                        "rmse",
                        "bias",
                        "empse",
                        "mrr_tir",
                        "mrr_tar180",
                        "mrr_tbr70",
                        "mrr_mage",
                        "mrr_cv",
                        "gmi_abserr",
                        "mean_abserr",
                        "group",
                        "pid",
                    )
                    if k in c
                }
                for c in cells
            ],
            "aggregate": {f"{m}|{me}|{p}": v for (m, me, p), v in agg.items()},
        }
        if skipped:
            out["skipped"] = skipped
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}: {len(cells)} cells", flush=True)

    # Checkpoint the in-proc results BEFORE the heavier/riskier PyPOTS stage so a
    # PyPOTS failure cannot lose the core 10-method benchmark.
    _write(skipped=["mrnn", "gpvae"] if args.pypots_python else ["mrnn", "gpvae"])

    if args.pypots_python and pypots_jobs:
        with tempfile.TemporaryDirectory() as tmpdir:
            for method in ("mrnn", "gpvae"):
                filled = _run_pypots(pypots_jobs, args.pypots_python, method, tmpdir)
                for rec, pred in zip(pypots_jobs, filled):
                    cells.append(
                        _score(
                            method,
                            rec["mech"],
                            rec["pct"],
                            np.asarray(pred, dtype=np.float32).reshape(-1),
                            rec["true"],
                            rec["tgt"],
                            std_mgdl,
                            mean_mgdl,
                            rec["meanfill"],
                            battery,
                            group=rec["group"],
                            pid=rec["pid"],
                        )
                    )
        _write(skipped=None)


def _score(
    name,
    mech,
    pct,
    pred,
    true_d2,
    tgt,
    std_mgdl,
    mean_mgdl,
    meanfill,
    battery,
    group=None,
    pid=None,
):
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    return {
        "method": name,
        "mech": mech,
        "pct": pct,
        "group": group,
        "pid": pid,
        "rmse": rmse_mgdl(pred, true_d2, tgt, std_mgdl),
        "bias": bias_mgdl(pred, true_d2, tgt, std_mgdl),
        "empse": empse_mgdl(pred, true_d2, tgt, std_mgdl),
        **downstream_summary(
            true_mgdl=true_d2 * std_mgdl + mean_mgdl,
            method_mgdl=pred * std_mgdl + mean_mgdl,
            meanfill_mgdl=meanfill,
            battery=battery,
        ),
    }


def _activity_trigger(samp, t_lo, t_hi, mm_activity):
    try:
        pid = int(samp.get("person_id"))
    except Exception:
        return None
    rec = mm_activity.get(pid)
    if rec is None:
        return None
    for ch in ("steps", "hr", "cal"):
        s = rec.get(ch)
        if s is not None and s.shape[0] >= t_hi:
            w = s[t_lo:t_hi].astype(np.float32)
            if np.nanstd(w) > 1e-6:
                return np.nan_to_num(w)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aireadi")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_participants", type=int, default=50)
    ap.add_argument(
        "--refine_ckpt_glob",
        default=os.path.join(ROOT, "method_checkpoints", "refine", "member*.pt"),
        help="Checkpoint glob for the CAIR ensemble.",
    )
    ap.add_argument("--n_refine_members", type=int, default=5)
    ap.add_argument("--mm_pkl", default=DEFAULT_MM)
    ap.add_argument(
        "--pypots_python",
        default=None,
        help="Path to isolated PyPOTS env python for MRNN/GP-VAE.",
    )
    ap.add_argument("--out", default=os.path.join(HERE, "toye_benchmark.json"))
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
