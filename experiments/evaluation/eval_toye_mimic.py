"""Toye MCAR/MAR/NMAR benchmark for MIMIC ICU vital signs."""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
import sys

for _p in (ROOT, os.path.join(ROOT, "baselines", "statistical")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cgm_datasets import load_dataset_splits
from utils.missingness_mechanisms import (
    create_mcar_mask,
    create_mar_mask,
    create_nmar_mask,
    native_gap_durations,
)
from experiments.evaluation.eval_toye_benchmark import (
    INPROC_BASELINES,
    PERCENTS,
    MECHS,
    _score,
    aggregate_cells,
    _run_pypots,
)
from experiments.evaluation.mimic_downstream import make_battery


def _make_masks_mimic(
    series, obs_mask, mech, pct, trigger, gap_durs, seed, mean, std, lo, hi
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
        mean_mgdl=mean,
        std_mgdl=std,
        lo_mgdl=lo,
        hi_mgdl=hi,
    )


def _covariate_trigger(samp, t_lo, t_hi, cov_slot):
    """MAR trigger from an in-record covariate vital (mm_<cov_slot>)."""
    key = f"mm_{cov_slot}"
    if key not in samp:
        return None
    s = np.asarray(samp[key], dtype=np.float32).reshape(-1)
    if s.shape[0] < t_hi:
        return None
    w = s[t_lo:t_hi]
    if np.nanstd(w) <= 1e-6:
        return None
    return np.nan_to_num(w)


def run(args):
    import torch

    device = (
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    _, _, test, meta = load_dataset_splits(args.dataset, prepared_dir=args.data_dir)
    test = test[: args.n_participants]
    std = float(meta.glucose_std_mgdl)
    mean = float(meta.glucose_mean_mgdl)
    t_lo, t_hi = meta.eval_window
    notes = meta.notes or {}
    lo = float(notes.get("nmar_lo", 65.0))
    hi = float(notes.get("nmar_hi", 100.0))
    cov_slot = str(notes.get("mar_covariate", "hr"))
    burden = notes.get(
        "burden", {"normal_lo": lo, "normal_hi": hi, "low_thr": lo, "high_thr": hi}
    )
    battery = make_battery(
        normal_lo=float(burden["normal_lo"]),
        normal_hi=float(burden["normal_hi"]),
        low_thr=float(burden["low_thr"]),
        high_thr=float(burden["high_thr"]),
    )

    cair = None
    if not args.skip_cair:
        from methods.cair import CAIRImputer

        rpaths = sorted(_glob.glob(args.cair_ckpt_glob))[: args.n_cair_members]
        assert rpaths, f"no CAIR checkpoints matched {args.cair_ckpt_glob}"
        cair = CAIRImputer(ckpt_paths=rpaths, device=device)

    cells = []
    pypots_jobs = []
    mar_skipped = 0
    default_durs = [3, 6, 9, 12] if meta.sampling_minutes == 5 else [1, 2, 3]
    for pi, samp in enumerate(test):
        careunit = str(samp.get("careunit", "UNK"))
        mortality = int(samp.get("in_hospital_mortality", 0) or 0)
        full_ts = samp["irg_ts"][:, 0].astype(np.float32)
        full_orig = samp["irg_ts_mask"][:, 0].astype(bool)
        d2 = samp["irg_ts"][t_lo:t_hi].copy()
        m2 = samp["irg_ts_mask"][t_lo:t_hi].copy()
        true_d2 = d2[:, 0].astype(np.float32)
        true_mgdl = true_d2 * std + mean
        gap_durs = native_gap_durations(samp["irg_ts_mask"]) or default_durs
        trigger = _covariate_trigger(samp, t_lo, t_hi, cov_slot)
        if trigger is None:
            mar_skipped += 1
        meanfill = np.full_like(true_mgdl, float(true_mgdl.mean()))

        for mech in MECHS:
            if mech == "mar" and trigger is None:
                continue
            for pct in PERCENTS:
                seed = 1000 * pi + 100 * MECHS.index(mech) + int(pct * 100)
                em, tm = _make_masks_mimic(
                    d2.copy(),
                    m2.copy(),
                    mech,
                    pct,
                    trigger,
                    gap_durs,
                    seed,
                    mean,
                    std,
                    lo,
                    hi,
                )
                obs = em[:, 0].astype(bool)
                tgt = tm[:, 0].astype(bool)
                if not tgt.any():
                    continue
                seen = np.where(obs, true_d2, np.nan)
                # Queue the identical masked series for the out-of-process PyPOTS
                # methods so MRNN/GP-VAE score on the same mask as the baselines.
                pypots_jobs.append(
                    {
                        "X": np.where(obs, true_d2, np.nan)
                        .reshape(-1, 1)
                        .astype(np.float32),
                        "mech": mech,
                        "pct": pct,
                        "pid": int(samp.get("subject_id", pi)),
                        "careunit": careunit,
                        "mortality": mortality,
                        "true": true_d2,
                        "tgt": tgt,
                        "meanfill": meanfill,
                    }
                )
                if not args.skip_baselines:
                    for name, fn in INPROC_BASELINES.items():
                        pred = fn(seen, obs.astype(np.float32))
                        cell = _score(
                            name,
                            mech,
                            pct,
                            pred,
                            true_d2,
                            tgt,
                            std,
                            mean,
                            meanfill,
                            battery,
                            group=careunit,
                            pid=int(samp.get("subject_id", pi)),
                        )
                        cell.update({"careunit": careunit, "mortality": mortality})
                        cells.append(cell)
                if cair is not None:
                    fe = full_orig.copy()
                    fe[t_lo:t_hi] = obs
                    fm = full_ts.copy()
                    fm[~fe] = 0.0
                    rpred = cair.impute(fm, fe.astype(np.float32))[t_lo:t_hi]
                    cell = _score(
                        "cair",
                        mech,
                        pct,
                        rpred,
                        true_d2,
                        tgt,
                        std,
                        mean,
                        meanfill,
                        battery,
                        group=careunit,
                        pid=int(samp.get("subject_id", pi)),
                    )
                    cell.update({"careunit": careunit, "mortality": mortality})
                    cells.append(cell)
        if pi % 25 == 0:
            print(f"  window {pi}/{len(test)}", flush=True)

    # Out-of-process PyPOTS methods (MRNN, GP-VAE): reuse the AI-READI driver's
    # _run_pypots helper so the isolated env is invoked identically. Each job's
    # masked series is the same NaN-at-deleted trace the in-proc baselines saw, so
    # scoring is on the identical mask. PYPOTS_EPOCHS is honored by the runner.
    if args.pypots_python and pypots_jobs:
        os.environ["PYPOTS_EPOCHS"] = str(args.pypots_epochs)
        with tempfile.TemporaryDirectory() as tmpdir:
            for method in ("mrnn", "gpvae"):
                filled = _run_pypots(pypots_jobs, args.pypots_python, method, tmpdir)
                for job, imputed in zip(pypots_jobs, filled):
                    cell = _score(
                        method,
                        job["mech"],
                        job["pct"],
                        np.asarray(imputed, dtype=np.float32)[:, 0],
                        job["true"],
                        job["tgt"],
                        std,
                        mean,
                        job["meanfill"],
                        battery,
                        group=job["careunit"],
                        pid=job["pid"],
                    )
                    cell.update(
                        {"careunit": job["careunit"], "mortality": job["mortality"]}
                    )
                    cells.append(cell)

    value_keys = [
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
    ]
    cell_keys = (
        ["method", "mech", "pct"] + value_keys + ["careunit", "mortality", "pid"]
    )
    # aggregate_cells returns tuple keys (method, mech, pct); stringify for JSON.
    aggregate = {
        f"{m}|{mech}|{pct}": v
        for (m, mech, pct), v in aggregate_cells(cells, value_keys).items()
    }
    out = {
        "dataset": meta.name,
        "n_participants": len(test),
        "mar_skipped_persons": mar_skipped,
        "units": notes.get("units", ""),
        "cells": [{k: c.get(k) for k in cell_keys} for c in cells],
        "aggregate": aggregate,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(
        f"wrote {args.out}: {len(cells)} cells, mar_skipped={mar_skipped}", flush=True
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mimic_abp")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_participants", type=int, default=400)
    ap.add_argument("--skip_cair", action="store_true")
    ap.add_argument(
        "--skip_baselines",
        action="store_true",
        help="Skip in-proc baselines (CAIR-only run; merge with a "
        "prior --skip_cair baseline JSON that used identical seeds).",
    )
    ap.add_argument(
        "--cair_ckpt_glob",
        default=os.path.join(
            ROOT, "method_checkpoints", "cair_mimic_abp", "uni_seed*.pt"
        ),
    )
    ap.add_argument("--n_cair_members", type=int, default=5)
    ap.add_argument(
        "--pypots_python",
        default=None,
        help="Path to isolated PyPOTS env python for MRNN/GP-VAE. "
        "If omitted those two methods are skipped.",
    )
    ap.add_argument(
        "--pypots_epochs",
        type=int,
        default=100,
        help="Training epochs for the PyPOTS MRNN/GP-VAE runners.",
    )
    ap.add_argument("--out", default=os.path.join(HERE, "toye_mimic.json"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
