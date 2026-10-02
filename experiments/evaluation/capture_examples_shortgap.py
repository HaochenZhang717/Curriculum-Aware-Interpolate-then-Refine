"""Capture per-window imputation TRAJECTORIES for the cross-dataset short-gap galleries (zero-shot REFINE vs classical interpolators on Ohio / HUPA-UCM / Shanghai)."""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATASET_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
for p in (ROOT, os.path.join(ROOT, "baselines", "statistical")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cgm_datasets import load_dataset_splits
from experiments.evaluation.eval_short_gaps import build_imputer, rmse_mgdl

# method -> the eval_short_gaps method key used to build its imputer
CLASSICAL = ["akima", "pchip", "linear", "spline", "savgol"]


def _build_methods(refine_ckpts, device):
    """Return {method_name: impute_fn(ts, mask)->pred}, plus ts_mods/static."""
    fns = {}
    # REFINE (published ensemble, zero-shot)
    ns = types.SimpleNamespace(
        method="refine", ckpt=None, ckpt_paths=",".join(refine_ckpts), ckpt_dir=None
    )
    rfn, ts_mods, static_blocks = build_imputer(ns, device)
    fns["refine"] = lambda ts, m, _f=rfn: _f(ts, m, None, None)
    # classical baselines (ignore modality/ctx)
    for name in CLASSICAL:
        ns = types.SimpleNamespace(
            method=name, ckpt=None, ckpt_paths=None, ckpt_dir=None
        )
        bfn, _, _ = build_imputer(ns, device)
        fns[name] = lambda ts, m, _f=bfn: _f(ts, m, None, None)
    return fns, ts_mods, static_blocks


def run(args):
    import torch

    device = (
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    train, val, test, meta = load_dataset_splits(
        args.dataset, prepared_dir=args.data_dir
    )
    if args.all_splits:
        test = list(train) + list(val) + list(test)
    test = test[: args.n_participants]
    std = float(meta.glucose_std_mgdl)
    mean = float(meta.glucose_mean_mgdl)
    t_lo, t_hi = meta.eval_window
    L_win = t_hi - t_lo
    gaps = [int(x) for x in args.gaps.split(",") if x.strip()]
    S = max(1, args.stride)
    ML = args.impute_max_len

    refine_ckpts = sorted(_glob.glob(args.refine_ckpt_glob))[: args.n_refine_members]
    assert refine_ckpts, f"no REFINE checkpoints matched {args.refine_ckpt_glob}"
    fns, ts_mods, static_blocks = _build_methods(refine_ckpts, device)
    from cgm_datasets.multimodal.modality_spec import build_mod_and_ctx

    methods = ["refine"] + CLASSICAL

    W_truth, W_obs, W_tgt = [], [], []
    W_pred = {m: [] for m in methods}
    win_meta = []

    for pi, samp in enumerate(test):
        full_ts = samp["irg_ts"][:, 0].astype(np.float32)
        full_orig = samp["irg_ts_mask"][:, 0].astype(bool)
        if full_orig.shape[0] < t_hi:
            continue
        mod_full, ctx_vec = build_mod_and_ctx(samp, ts_mods, static_blocks)
        truth_win = full_ts[t_lo:t_hi].astype(np.float32)
        orig_win = full_orig[t_lo:t_hi]

        for Lg in gaps:
            rng = np.random.default_rng(1000 * pi + Lg)
            score_off = list(range(0, Lg, S))
            cands = [
                s
                for s in range(max(t_lo, S), t_hi - Lg)
                if full_orig[s - S]
                and full_orig[s + Lg]
                and all(full_orig[s + j] for j in score_off)
            ]
            if not cands:
                continue
            picks = rng.choice(
                len(cands), size=min(args.n_placements, len(cands)), replace=False
            )
            for idx in picks:
                s = cands[int(idx)]
                fe = full_orig.copy()
                fe[s : s + Lg] = False
                fm = full_ts.copy()
                fm[~fe] = 0.0
                if ML and len(fm) > ML:
                    fm_i, fe_i = fm[:ML], fe[:ML]
                    mod_i = mod_full[:ML] if mod_full is not None else None
                else:
                    fm_i, fe_i, mod_i = fm, fe, mod_full

                carved_rel = np.zeros(L_win, bool)
                carved_rel[s - t_lo : s - t_lo + Lg] = True
                tgt_rel = np.zeros(L_win, bool)
                for j in score_off:
                    tgt_rel[s - t_lo + j] = True
                obs_rel = orig_win & ~carved_rel

                preds_this = {}
                for m in methods:
                    predf = np.asarray(fns[m](fm_i, fe_i)).ravel()
                    # pad back if impute_max_len truncated below t_hi (never for day-2)
                    if predf.shape[0] < t_hi:
                        tmp = full_ts.copy().astype(np.float32)
                        tmp[: predf.shape[0]] = predf
                        predf = tmp
                    preds_this[m] = predf[t_lo:t_hi].astype(np.float32)

                score_idx = np.where(tgt_rel)[0]
                rmse = {
                    m: float(
                        rmse_mgdl(preds_this[m][score_idx], truth_win[score_idx], std)
                    )
                    for m in methods
                }

                w = len(win_meta)
                _pid = samp.get("person_id", pi)
                _pid = int(_pid) if isinstance(_pid, (int, np.integer)) else str(_pid)
                W_truth.append(truth_win.copy())
                W_obs.append(obs_rel.copy())
                W_tgt.append(tgt_rel.copy())
                for m in methods:
                    W_pred[m].append(preds_this[m])
                win_meta.append(
                    {
                        "idx": w,
                        "pid": _pid,
                        "gap_len_min": int(Lg * 5),
                        "mech": f"gap{Lg*5}min",
                        "pct": float(Lg * 5),
                        "n_target": int(tgt_rel.sum()),
                        "gap_spans": [[int(s - t_lo), int(s - t_lo + Lg - 1)]],
                        "rmse": rmse,
                    }
                )
        if pi % 10 == 0:
            print(f"  pid {pi}/{len(test)} (captured {len(win_meta)})", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    arrays = {
        "truth": np.stack(W_truth).astype(np.float32),
        "obs": np.stack(W_obs).astype(bool),
        "tgt": np.stack(W_tgt).astype(bool),
    }
    for m in methods:
        arrays[f"pred_{m}"] = np.stack(W_pred[m]).astype(np.float32)
    np.savez_compressed(args.out + ".npz", **arrays)

    meta_out = {
        "dataset": meta.name,
        "display_name": getattr(meta, "display_name", meta.name),
        "units": "mg/dL",
        "glucose_mean_mgdl": mean,
        "glucose_std_mgdl": std,
        "sampling_minutes": float(meta.sampling_minutes),
        "stride": S,
        "eval_window": [int(t_lo), int(t_hi)],
        "L": int(L_win),
        "methods": methods,
        "gaps_min": [Lg * 5 for Lg in gaps],
        "n_participants": len(test),
        "refine_ckpt_glob": args.refine_ckpt_glob,
        "windows": win_meta,
    }
    json.dump(meta_out, open(args.out + "_meta.json", "w"), indent=2)
    print(
        f"wrote {args.out}.npz ({len(win_meta)} windows, {len(methods)} methods)",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--all_splits", action="store_true", default=True)
    ap.add_argument("--no_all_splits", dest="all_splits", action="store_false")
    ap.add_argument("--gaps", default="3,6,9,12", help="gap lengths in 5-min samples")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--impute_max_len", type=int, default=0)
    ap.add_argument("--n_participants", type=int, default=40)
    ap.add_argument("--n_placements", type=int, default=3)
    ap.add_argument("--refine_ckpt_glob", required=True)
    ap.add_argument("--n_refine_members", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
