"""Capture per-window imputation TRAJECTORIES for the Toye galleries."""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATASET_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
for _p in (ROOT, os.path.join(ROOT, "baselines", "statistical")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cgm_datasets import load_dataset_splits
from utils.missingness_mechanisms import native_gap_durations
from experiments.evaluation.eval_toye_benchmark import (
    INPROC_BASELINES,
    MECHS,
    _run_pypots,
    _make_masks,
    load_mm_activity,
    load_group_map,
    _activity_trigger,
)
from experiments.evaluation.eval_toye_mimic import _covariate_trigger, _make_masks_mimic
from experiments.evaluation.toye_metrics import rmse_mgdl

DEFAULT_MM = os.path.join(
    DATASET_ROOT, "trace", "diffusion", "data", "aireadi_cgm_mm", "aireadi_cgm_test.pkl"
)


def gap_spans(mask: np.ndarray):
    """Contiguous [start, end] index spans (inclusive) where mask is True."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    spans, s = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            spans.append([int(s), int(a)])
            s = b
    spans.append([int(s), int(idx[-1])])
    return spans


def run(args):
    import torch

    device = (
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    is_mimic = args.dataset.startswith("mimic")

    _, _, test, meta = load_dataset_splits(args.dataset, prepared_dir=args.data_dir)
    test = test[: args.n_participants]
    std = float(meta.glucose_std_mgdl)
    mean = float(meta.glucose_mean_mgdl)
    t_lo, t_hi = meta.eval_window
    L = t_hi - t_lo
    dt = float(meta.sampling_minutes)
    notes = meta.notes or {}
    units = notes.get("units", "mg/dL")

    if is_mimic:
        lo = float(notes.get("nmar_lo", 65.0))
        hi = float(notes.get("nmar_hi", 100.0))
        cov_slot = str(notes.get("mar_covariate", "hr"))
        default_durs = [3, 6, 9, 12] if meta.sampling_minutes == 5 else [1, 2, 3]
    else:
        mm_activity = load_mm_activity(args.mm_pkl)
        group_map = load_group_map()

    rates = [float(x) for x in args.rates.split(",")]

    refine = None
    if not args.skip_refine:
        from methods.refine import RefineImputer

        rpaths = sorted(_glob.glob(args.refine_ckpt_glob))[: args.n_refine_members]
        assert rpaths, f"no REFINE checkpoints matched {args.refine_ckpt_glob}"
        refine = RefineImputer(ckpt_paths=rpaths, device=device)

    methods = list(INPROC_BASELINES.keys()) + (["refine"] if refine is not None else [])
    pypots_methods = ["mrnn", "gpvae"] if args.pypots_python else []
    all_methods = methods + pypots_methods

    # per-window records; preds stored as (W, L) stacks at the end
    W_truth, W_obs, W_tgt = [], [], []
    W_pred = {m: [] for m in all_methods}
    win_meta = []
    pypots_jobs = []
    mar_skipped = 0

    for pi, samp in enumerate(test):
        full_ts = samp["irg_ts"][:, 0].astype(np.float32)
        full_orig = samp["irg_ts_mask"][:, 0].astype(bool)
        d2 = samp["irg_ts"][t_lo:t_hi].copy()
        m2 = samp["irg_ts_mask"][t_lo:t_hi].copy()
        true_d2 = d2[:, 0].astype(np.float32)
        gap_durs = native_gap_durations(samp["irg_ts_mask"]) or (
            default_durs if is_mimic else [3, 6, 9, 12]
        )

        if is_mimic:
            pid = int(samp.get("subject_id", pi))
            group = str(samp.get("careunit", "UNK"))
            mortality = int(samp.get("in_hospital_mortality", 0) or 0)
            trigger = _covariate_trigger(samp, t_lo, t_hi, cov_slot)
        else:
            pid = int(samp["person_id"])
            group = group_map.get(pid, "unknown")
            mortality = None
            trigger = _activity_trigger(samp, t_lo, t_hi, mm_activity)
        if trigger is None:
            mar_skipped += 1

        for mech in MECHS:
            if mech == "mar" and trigger is None:
                continue
            for pct in rates:
                seed = 1000 * pi + 100 * MECHS.index(mech) + int(pct * 100)
                if is_mimic:
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
                else:
                    em, tm = _make_masks(
                        d2.copy(),
                        m2.copy(),
                        mech,
                        pct,
                        trigger,
                        gap_durs,
                        seed,
                        mean,
                        std,
                    )
                obs = em[:, 0].astype(bool)
                tgt = tm[:, 0].astype(bool)
                if not tgt.any():
                    continue
                seen = np.where(obs, true_d2, np.nan)

                w = len(win_meta)  # this window's index
                preds_this = {}
                for name, fn in INPROC_BASELINES.items():
                    preds_this[name] = np.asarray(
                        fn(seen, obs.astype(np.float32)), dtype=np.float32
                    ).reshape(-1)
                if refine is not None:
                    fe = full_orig.copy()
                    fe[t_lo:t_hi] = obs
                    fm = full_ts.copy()
                    fm[~fe] = 0.0
                    preds_this["refine"] = np.asarray(
                        refine.impute(fm, fe.astype(np.float32)), dtype=np.float32
                    )[t_lo:t_hi]

                rmse = {
                    m: float(rmse_mgdl(preds_this[m], true_d2, tgt, std))
                    for m in methods
                }

                W_truth.append(true_d2.copy())
                W_obs.append(obs.copy())
                W_tgt.append(tgt.copy())
                for m in methods:
                    W_pred[m].append(preds_this[m])
                for m in pypots_methods:
                    W_pred[m].append(np.full(L, np.nan, np.float32))  # filled later
                win_meta.append(
                    {
                        "idx": w,
                        "pid": pid,
                        "mech": mech,
                        "pct": float(pct),
                        "group": group,
                        "mortality": mortality,
                        "n_target": int(tgt.sum()),
                        "gap_spans": gap_spans(tgt),
                        "rmse": rmse,
                    }
                )
                if pypots_methods:
                    pypots_jobs.append(
                        {
                            "w": w,
                            "X": seen.reshape(-1, 1).astype(np.float32),
                            "true": true_d2,
                            "tgt": tgt,
                        }
                    )
        if pi % 10 == 0:
            print(f"  window {pi}/{len(test)}  (captured {len(win_meta)})", flush=True)

    # out-of-process PyPOTS (MRNN / GP-VAE): identical NaN-at-deleted trace -
    # Each method is isolated: a node/env failure (e.g. GP-VAE Triton JIT running
    # /tmp out of space) leaves that method as NaN but never loses the whole run.
    pypots_done = []
    if pypots_methods and pypots_jobs:
        os.environ["PYPOTS_EPOCHS"] = str(args.pypots_epochs)
        with tempfile.TemporaryDirectory() as tmpdir:
            for method in pypots_methods:
                try:
                    filled = _run_pypots(
                        pypots_jobs, args.pypots_python, method, tmpdir
                    )
                except Exception as e:
                    print(
                        f"[warn] pypots {method} failed ({e}); leaving NaN", flush=True
                    )
                    continue
                for job, imputed in zip(pypots_jobs, filled):
                    series = np.asarray(imputed, dtype=np.float32).reshape(-1)
                    w = job["w"]
                    W_pred[method][w] = series
                    win_meta[w]["rmse"][method] = float(
                        rmse_mgdl(series, job["true"], job["tgt"], std)
                    )
                pypots_done.append(method)
    all_methods = methods + pypots_done  # meta advertises only what succeeded

    # - serialize -
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    arrays = {
        "truth": np.stack(W_truth).astype(np.float32),
        "obs": np.stack(W_obs).astype(bool),
        "tgt": np.stack(W_tgt).astype(bool),
    }
    for m in all_methods:
        arrays[f"pred_{m}"] = np.stack(W_pred[m]).astype(np.float32)
    np.savez_compressed(args.out + ".npz", **arrays)

    meta_out = {
        "dataset": meta.name,
        "units": units,
        "glucose_mean_mgdl": mean,
        "glucose_std_mgdl": std,
        "sampling_minutes": dt,
        "eval_window": [int(t_lo), int(t_hi)],
        "L": int(L),
        "methods": all_methods,
        "rates": rates,
        "mechs": MECHS,
        "n_participants": len(test),
        "mar_skipped_persons": mar_skipped,
        "refine_ckpt_glob": args.refine_ckpt_glob,
        "windows": win_meta,
    }
    json.dump(meta_out, open(args.out + "_meta.json", "w"), indent=2)
    print(
        f"wrote {args.out}.npz  ({len(win_meta)} windows, "
        f"{len(all_methods)} methods, L={L})",
        flush=True,
    )
    print(f"wrote {args.out}_meta.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mimic_abp")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_participants", type=int, default=96)
    ap.add_argument(
        "--rates",
        default="0.15,0.30",
        help="comma-separated missingness rates to capture",
    )
    ap.add_argument("--skip_refine", action="store_true")
    ap.add_argument(
        "--refine_ckpt_glob",
        required=True,
        help="Glob for REFINE ensemble checkpoints (absolute or repo-relative).",
    )
    ap.add_argument("--n_refine_members", type=int, default=5)
    ap.add_argument(
        "--mm_pkl",
        default=DEFAULT_MM,
        help="AI-READI wearable pickle for the MAR trigger (aireadi only).",
    )
    ap.add_argument(
        "--pypots_python",
        default=None,
        help="Isolated PyPOTS env python for MRNN/GP-VAE (omit to skip).",
    )
    ap.add_argument("--pypots_epochs", type=int, default=100)
    ap.add_argument(
        "--out", default=os.path.join(ROOT, "results", "examples", "toye_examples")
    )
    args = ap.parse_args()
    # allow repo-relative globs
    if not os.path.isabs(args.refine_ckpt_glob) and not _glob.glob(
        args.refine_ckpt_glob
    ):
        cand = os.path.join(ROOT, args.refine_ckpt_glob)
        if _glob.glob(cand):
            args.refine_ckpt_glob = cand
    run(args)


if __name__ == "__main__":
    main()
