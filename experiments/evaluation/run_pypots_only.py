#!/usr/bin/env python3
"""Run only the PyPOTS methods for the Toye benchmark."""

from __future__ import annotations

from utils.paths import DATA_ROOT, REPO_ROOT, CACHE_ROOT

import argparse
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cgm_datasets import load_dataset_splits
from experiments.evaluation.eval_toye_benchmark import (
    DEFAULT_MM,
    MECHS,
    PERCENTS,
    ROOT as EVAL_ROOT,
    _activity_trigger,
    _make_masks,
    _run_pypots,
    _score,
    load_battery,
    load_group_map,
    load_mm_activity,
    native_gap_durations,
)

if EVAL_ROOT not in sys.path:
    sys.path.insert(0, EVAL_ROOT)

VALUE_KEYS = (
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


def run(args):
    _, _, test, meta = load_dataset_splits(args.dataset)
    test = test[: args.n_participants]
    std_mgdl = float(meta.glucose_std_mgdl)
    mean_mgdl = float(meta.glucose_mean_mgdl)
    t_lo, t_hi = meta.eval_window
    mm_activity = load_mm_activity(DEFAULT_MM)
    group_map = load_group_map()
    battery = load_battery()

    pypots_jobs = []
    cells = []
    for pi, samp in enumerate(test):
        pid = int(samp["person_id"])
        grp = group_map.get(pid, "unknown")
        d2 = samp["irg_ts"][t_lo:t_hi].copy()
        m2 = samp["irg_ts_mask"][t_lo:t_hi].copy()
        true_d2 = d2[:, 0].astype(np.float32)
        true_mgdl = true_d2 * std_mgdl + mean_mgdl
        gap_durs = native_gap_durations(samp["irg_ts_mask"]) or [3, 6, 9, 12]
        trigger = _activity_trigger(samp, t_lo, t_hi, mm_activity)
        meanfill = np.full_like(true_mgdl, float(true_mgdl.mean()))

        for mech in MECHS:
            if mech == "mar" and trigger is None:
                continue
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

    if pypots_jobs:
        # Participants' day-2 windows are not all the same length (a few are
        # shorter than t_hi-t_lo), but PyPOTS batches a fixed n_steps. Pad every
        # masked series to the common max length with NaN (treated as missing),
        # impute, then slice each prediction back to its true length for scoring.
        lmax = max(job["X"].shape[0] for job in pypots_jobs)

        def _pad(x):
            if x.shape[0] == lmax:
                return x
            pad = np.full((lmax - x.shape[0], x.shape[1]), np.nan, dtype=x.dtype)
            return np.concatenate([x, pad], axis=0)

        padded_records = [{"X": _pad(job["X"])} for job in pypots_jobs]
        with tempfile.TemporaryDirectory() as tmpdir:
            for method in ("mrnn", "gpvae"):
                filled = _run_pypots(padded_records, args.pypots_python, method, tmpdir)
                for job, pred in zip(pypots_jobs, filled):
                    tlen = job["true"].shape[0]
                    cells.append(
                        _score(
                            method,
                            job["mech"],
                            job["pct"],
                            np.asarray(pred).reshape(-1)[:tlen],
                            job["true"],
                            job["tgt"],
                            std_mgdl,
                            mean_mgdl,
                            job["meanfill"],
                            battery,
                            group=job["group"],
                            pid=job["pid"],
                        )
                    )

    out = {
        "dataset": meta.name,
        "n_participants": len(test),
        "cells": [{k: c[k] for k in VALUE_KEYS if k in c} for c in cells],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aireadi")
    ap.add_argument("--n_participants", type=int, default=352)
    ap.add_argument(
        "--pypots_python",
        default=os.environ.get("PYPOTS_PYTHON", __import__("sys").executable),
    )
    ap.add_argument("--out", default=os.path.join(HERE, "toye_n352_pypots.json"))
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
