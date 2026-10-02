#!/usr/bin/env python3
"""Reproduce the CAIR headline on AI-READI from the clean methods/ package."""

from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.physiological_masking import create_physiological_mask
from methods.cair import CAIRImputer
from cgm_datasets import load_dataset_splits

CGM_STD = 42.33
T_LO, T_HI = 288, 576
STRATS = ["meal_post", "sleep", "ascending", "dipping", "combined"]


def rmse(pred, true, m, cgm_std=CGM_STD):
    m = np.asarray(m).astype(bool)
    if not m.any():
        return float("nan")
    e = (pred[m] - true[m]) * cgm_std
    return float(np.sqrt((e**2).mean()))


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_participants", type=int, default=50)
    ap.add_argument("--n_masks", type=int, default=5)
    ap.add_argument(
        "--ckpt_dir", default=os.path.join(ROOT, "method_checkpoints", "cair")
    )
    ap.add_argument(
        "--ckpt",
        default=None,
        help="Single checkpoint path; creates a 1-model ensemble (ablation mode). "
        "Overrides --ckpt_dir when provided.",
    )
    ap.add_argument(
        "--ckpt_paths",
        default=None,
        help="Comma-separated checkpoint paths (a rung's 5 members) to ensemble.",
    )
    ap.add_argument(
        "--dataset", default="aireadi", help="Dataset registry key (default: aireadi)."
    )
    ap.add_argument(
        "--data_dir", default=None, help="Prepared dataset directory override."
    )
    ap.add_argument(
        "--n_refinements",
        type=int,
        default=2,
        help="CAIR inference refinement passes (ablation knob; publish default=2).",
    )
    ap.add_argument("--out", default=os.path.join(HERE, "cair_eval.json"))
    if args is None:
        args = ap.parse_args()
    import torch

    device = (
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )

    _, _, test, metadata = load_dataset_splits(args.dataset, prepared_dir=args.data_dir)
    test = test[: args.n_participants]
    cgm_std = float(metadata.glucose_std_mgdl)
    t_lo, t_hi = metadata.eval_window
    nref = int(args.n_refinements)
    if args.ckpt_paths:
        paths = [p for p in args.ckpt_paths.split(",") if p]
        imp = CAIRImputer(ckpt_paths=paths, device=device, n_refinements=nref)
        spec_ckpt = paths[0]
    elif args.ckpt is not None:
        imp = CAIRImputer(ckpt_paths=[args.ckpt], device=device, n_refinements=nref)
        spec_ckpt = args.ckpt
    else:
        import glob as _glob

        imp = CAIRImputer(ckpt_dir=args.ckpt_dir, device=device, n_refinements=nref)
        spec_ckpt = sorted(_glob.glob(os.path.join(args.ckpt_dir, "*.pt")))[0]

    # per-record modality inputs (same spec helper as training -> identical layout).
    from cgm_datasets.multimodal.modality_spec import build_mod_and_ctx

    _cfg = torch.load(spec_ckpt, map_location="cpu", weights_only=False).get(
        "config", {}
    )
    ts_mods = _cfg.get("ts_mods", []) or []
    static_blocks = _cfg.get("static_blocks", []) or []
    print(f"[eval] ts_mods={ts_mods} static_blocks={static_blocks}")

    rm = {s: [] for s in STRATS}
    t0 = time.time()
    for pi, samp in enumerate(test):
        if pi % 10 == 0:
            print(f"  pid {pi}/{len(test)} ({time.time()-t0:.0f}s)", flush=True)
        full_ts = samp["irg_ts"][:, 0].astype(np.float32)
        full_orig = samp["irg_ts_mask"][:, 0].astype(bool)
        d2 = samp["irg_ts"][t_lo:t_hi].copy()
        m2 = samp["irg_ts_mask"][t_lo:t_hi].copy()
        true_d2 = d2[:, 0].astype(np.float32)
        mod_full, ctx_vec = build_mod_and_ctx(samp, ts_mods, static_blocks)
        for strat in STRATS:
            for mi in range(args.n_masks):
                seed = 100 * pi + 10 * STRATS.index(strat) + mi
                em, tm = create_physiological_mask(
                    d2.copy(),
                    m2.copy(),
                    strategy=strat,
                    target_ratio=0.20,
                    seed=int(seed),
                )
                obs = em[:, 0].astype(bool)
                tgt = tm[:, 0].astype(bool)
                if not tgt.any():
                    continue
                fe = full_orig.copy()
                fe[t_lo:t_hi] = obs
                fm = full_ts.copy()
                fm[~fe] = 0.0
                pred = imp.impute(
                    fm, fe.astype(np.float32), modality=mod_full, ctx_vec=ctx_vec
                )[t_lo:t_hi]
                rm[strat].append(rmse(pred, true_d2, tgt, cgm_std))

    per = {s: float(np.nanmean(rm[s])) if rm[s] else float("nan") for s in STRATS}
    allv = [v for s in STRATS for v in rm[s] if v == v]
    avg = float(np.mean(allv)) if allv else float("nan")
    print(f"\nCAIR physiological evaluation: {metadata.display_name}")
    for s in STRATS:
        print(f"  {s:10s} {per[s]:.2f}")
    print(f"  {'AVG':10s} {avg:.2f} mg/dL")
    json.dump(
        {
            "per_strategy": per,
            "avg": avg,
            "n_participants": len(test),
            "n_masks": args.n_masks,
            "dataset": metadata.name,
        },
        open(args.out, "w"),
        indent=2,
    )
    print(f"Saved -> {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
