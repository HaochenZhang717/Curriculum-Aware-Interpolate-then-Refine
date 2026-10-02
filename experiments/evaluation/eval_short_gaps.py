#!/usr/bin/env python3
"""Short-range gap imputation analysis: RMSE stratified by GAP LENGTH."""

from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, os.path.join(ROOT, "baselines", "statistical")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cgm_datasets import load_dataset_splits

CGM_STD = 42.33


def rmse_mgdl(pred, true, cgm_std):
    e = (np.asarray(pred) - np.asarray(true)) * cgm_std
    return float(np.sqrt((e**2).mean()))


def build_imputer(args, device):
    """Return (impute(ts, m, mod, ctx) -> pred, ts_mods, static_blocks).

    Baselines ignore mod/ctx and return empty modality specs.
    """
    method = args.method
    if method == "cair":
        import torch
        from methods.cair import CAIRImputer

        nref = int(args.n_refinements)
        if args.ckpt_paths:
            paths = [p for p in args.ckpt_paths.split(",") if p]
            imp = CAIRImputer(ckpt_paths=paths, device=device, n_refinements=nref)
            spec_ckpt = paths[0]
        elif args.ckpt:
            imp = CAIRImputer(ckpt_paths=[args.ckpt], device=device, n_refinements=nref)
            spec_ckpt = args.ckpt
        else:
            import glob as _glob

            imp = CAIRImputer(ckpt_dir=args.ckpt_dir, device=device, n_refinements=nref)
            spec_ckpt = sorted(_glob.glob(os.path.join(args.ckpt_dir, "*.pt")))[0]
        cfg = torch.load(spec_ckpt, map_location="cpu", weights_only=False).get(
            "config", {}
        )
        ts_mods = cfg.get("ts_mods", []) or []
        static_blocks = cfg.get("static_blocks", []) or []
        fn = lambda ts, m, mod=None, ctx=None: np.asarray(
            imp.impute(ts, m.astype(np.float32), modality=mod, ctx_vec=ctx)
        ).ravel()
        return fn, ts_mods, static_blocks
    if method in (
        "pchip",
        "linear",
        "akima",
        "forward_fill",
        "spline",
        "savgol",
        "ewma",
        "ar",
    ):
        if method == "pchip":
            from baselines_autoregressive import pchip_impute as bfn
        elif method == "ar":
            from baselines_autoregressive import ar_bidirectional as bfn
        elif method == "linear":
            from simple import linear_interp as bfn
        elif method == "forward_fill":
            from simple import forward_fill as bfn
        elif method == "akima":
            from sota_baselines import akima_interp as bfn
        elif method == "spline":
            from sota_baselines import spline_cubic as bfn
        elif method == "savgol":
            from sota_baselines import savitzky_golay as bfn
        else:  # ewma
            from sota_baselines import ewma_impute as bfn
        # ewma_impute indexes series[i, 0] -> needs a (T, 1) input; the others take (T,).
        if method == "ewma":
            return (
                (
                    lambda ts, m, mod=None, ctx=None: np.asarray(
                        bfn(np.asarray(ts).reshape(-1, 1), np.asarray(m).reshape(-1, 1))
                    ).ravel()
                ),
                [],
                [],
            )
        return (
            (lambda ts, m, mod=None, ctx=None: np.asarray(bfn(ts, m)).ravel()),
            [],
            [],
        )
    raise ValueError(f"unknown method {method!r}")


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method",
        required=True,
        choices=[
            "cair",
            "pchip",
            "linear",
            "akima",
            "forward_fill",
            "spline",
            "savgol",
            "ewma",
            "ar",
        ],
    )
    ap.add_argument("--ckpt", default=None, help="single CAIR ckpt (method=cair)")
    ap.add_argument(
        "--ckpt_paths",
        default=None,
        help="comma-separated CAIR ckpts (a rung's 5 members) to ensemble",
    )
    ap.add_argument(
        "--ckpt_dir",
        default=os.path.join(ROOT, "method_checkpoints", "cair"),
        help="CAIR ensemble dir (method=cair, when --ckpt omitted)",
    )
    ap.add_argument("--dataset", default="aireadi")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument(
        "--all_splits",
        action="store_true",
        help="Pool train+val+test as the eval set. Valid for zero-shot transfer "
        "(no target data trained the model); maximizes power on small external cohorts.",
    )
    ap.add_argument(
        "--gaps",
        default="3,6,9,12",
        help="gap lengths in 5-min samples (3=15min..12=60min)",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="native cadence of the dataset in 5-min slots (1=5-min CGM; 3=15-min CGM like "
        "Shanghai). Gaps stay defined in wall-clock minutes but are carved/scored at the "
        "native cadence, so RMSE is measured only where real ground truth exists.",
    )
    ap.add_argument("--n_participants", type=int, default=50)
    ap.add_argument(
        "--n_placements",
        type=int,
        default=10,
        help="random gap placements per (participant,L)",
    )
    ap.add_argument(
        "--impute_max_len",
        type=int,
        default=0,
        help="Truncate the series passed to the imputer to the first N samples (0=off). "
        "Provably lossless for the day-2 eval window when N >= t_hi + model_window "
        "(=1152 with window 576), and avoids imputing many wasted days on long "
        "external series (e.g. OhioT1DM ~52 days).",
    )
    ap.add_argument(
        "--n_refinements",
        type=int,
        default=2,
        help="CAIR inference refinement passes (ablation knob; publish default=2).",
    )
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    if args is None:
        args = ap.parse_args()

    device = "cpu"
    if args.method == "cair":
        import torch

        device = (
            f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
        )

    train, val, test, metadata = load_dataset_splits(
        args.dataset, prepared_dir=args.data_dir
    )
    if args.all_splits:
        test = list(train) + list(val) + list(test)
    test = test[: args.n_participants]
    cgm_std = float(metadata.glucose_std_mgdl)
    t_lo, t_hi = metadata.eval_window
    gaps = [int(x) for x in args.gaps.split(",") if x.strip()]
    impute, ts_mods, static_blocks = build_imputer(args, device)
    from cgm_datasets.multimodal.modality_spec import build_mod_and_ctx

    print(f"[eval_short] ts_mods={ts_mods} static_blocks={static_blocks}")

    per_gap = {L: [] for L in gaps}
    n_skipped = {L: 0 for L in gaps}
    t0 = time.time()
    for pi, samp in enumerate(test):
        if pi % 10 == 0:
            print(f"  pid {pi}/{len(test)} ({time.time()-t0:.0f}s)", flush=True)
        full_ts = samp["irg_ts"][:, 0].astype(np.float32)
        full_orig = samp["irg_ts_mask"][:, 0].astype(bool)
        if full_orig.shape[0] < t_hi:  # no full day-2 window -> skip participant
            for L in gaps:
                n_skipped[L] += 1
            continue
        mod_full, ctx_vec = build_mod_and_ctx(samp, ts_mods, static_blocks)
        S = max(1, args.stride)
        for L in gaps:
            rng = np.random.default_rng(1000 * pi + L)
            score_off = list(
                range(0, L, S)
            )  # observed sub-positions inside the wall-clock gap
            # candidate gap starts whose interior sub-positions + both native boundaries are observed
            cands = [
                s
                for s in range(max(t_lo, S), t_hi - L)
                if full_orig[s - S]
                and full_orig[s + L]
                and all(full_orig[s + j] for j in score_off)
            ]
            if not cands:
                n_skipped[L] += 1
                continue
            picks = rng.choice(
                len(cands), size=min(args.n_placements, len(cands)), replace=False
            )
            ML = args.impute_max_len
            for idx in picks:
                s = cands[int(idx)]
                fe = full_orig.copy()
                fe[s : s + L] = False  # carve the contiguous gap
                fm = full_ts.copy()
                fm[~fe] = 0.0
                if ML and len(fm) > ML:  # bound impute cost (lossless for day-2)
                    fm_i, fe_i = fm[:ML], fe[:ML]
                    mod_i = mod_full[:ML] if mod_full is not None else None
                else:
                    fm_i, fe_i, mod_i = fm, fe, mod_full
                predf = np.asarray(impute(fm_i, fe_i, mod_i, ctx_vec))
                idxs = [
                    s + j for j in score_off
                ]  # score only where ground truth exists
                per_gap[L].append(rmse_mgdl(predf[idxs], full_ts[idxs], cgm_std))

    summary = {
        f"{L*5}min": (float(np.mean(per_gap[L])) if per_gap[L] else float("nan"))
        for L in gaps
    }
    counts = {f"{L*5}min": len(per_gap[L]) for L in gaps}
    label = (
        args.ckpt
        and os.path.basename(args.ckpt)
        or (args.method if args.method != "cair" else "ensemble")
    )
    print(f"\n=== short-gap RMSE (mg/dL) | method={args.method} | {label} ===")
    for L in gaps:
        print(
            f"  {L*5:3d}min  RMSE={summary[f'{L*5}min']:.2f}  (n={counts[f'{L*5}min']})"
        )
    out = {
        "method": args.method,
        "label": label,
        "by_gap_min": summary,
        "n_eval": counts,
        "n_skipped_participants": n_skipped,
        "n_participants": len(test),
        "n_placements": args.n_placements,
        "stride": int(max(1, args.stride)),
        "all_splits": bool(args.all_splits),
        "impute_max_len": int(args.impute_max_len),
        "n_refinements": int(args.n_refinements),
        "ckpt_paths": args.ckpt_paths,
        "dataset": metadata.name,
        "ckpt": args.ckpt,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Saved -> {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
