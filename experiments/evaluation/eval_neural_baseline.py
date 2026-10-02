#!/usr/bin/env python3
"""Fill the SAITS/BRITS rows of the REFINE paper's Tables 1-2 on AI-READI."""

from __future__ import annotations

from utils.paths import DATA_ROOT, REPO_ROOT, CACHE_ROOT
import argparse, json, os, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.physiological_masking import create_physiological_mask
from cgm_datasets import load_dataset_splits

STRATS = ["meal_post", "sleep", "ascending", "dipping", "combined"]
DEFAULT_PYPOTS = os.environ.get("PYPOTS_PYTHON", __import__("sys").executable)
# PyPOTS site-packages live on shared storage even though the env's python is a
# /home symlink (unusable on isolated compute nodes). We drive them with any
# py3.10 interpreter by injecting this on the subprocess PYTHONPATH only.
DEFAULT_PYPOTS_SITE = os.environ.get("PYPOTS_SITE", "")


def rmse_mgdl(pred, true, cgm_std):
    e = (np.asarray(pred) - np.asarray(true)) * cgm_std
    return float(np.sqrt((e**2).mean()))


def run_pypots(windows, method, pypots_python, epochs, pypots_site=None):
    """windows: (N, L) float array, NaN at non-observed. Returns imputed (N, L)."""
    X = np.asarray(windows, dtype=np.float32)[:, :, None]  # (N, L, 1)
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "in.npz")
        out = os.path.join(tmp, "out.npz")
        np.savez(inp, X=X)
        runner = os.path.join(ROOT, "baselines", "toye", "pypots_runner.py")
        env = dict(os.environ, PYPOTS_EPOCHS=str(epochs))
        if pypots_site:  # subprocess-only: never touches the outer interpreter
            env["PYTHONPATH"] = pypots_site + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(
            [
                pypots_python,
                runner,
                "--in",
                inp,
                "--out",
                out,
                "--method",
                method,
                "--epochs",
                str(epochs),
            ],
            check=True,
            env=env,
        )
        return np.load(out)["X_imputed"][:, :, 0]  # (N, L)


def collect_phys(test, t_lo, t_hi, n_masks):
    """Replicate eval_refine.py masking. Returns (windows, records, n_skipped).
    record = (strategy, target_index_array, true_values_at_target). Participants
    without a full day-2 window are skipped (same rule as eval_short_gaps.py; the
    fixed-length batch requires uniform 288-sample windows)."""
    windows, records = [], []
    n_skipped = 0
    for pi, samp in enumerate(test):
        if samp["irg_ts"].shape[0] < t_hi:
            n_skipped += 1
            continue
        d2 = samp["irg_ts"][t_lo:t_hi].copy()
        m2 = samp["irg_ts_mask"][t_lo:t_hi].copy()
        true_d2 = d2[:, 0].astype(np.float32)
        for strat in STRATS:
            for mi in range(n_masks):
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
                win = np.where(obs, true_d2, np.nan).astype(np.float32)
                windows.append(win)
                records.append((strat, np.flatnonzero(tgt), true_d2[tgt]))
    return np.stack(windows, 0), records, n_skipped


def collect_shortgap(test, t_lo, t_hi, gaps, stride, n_placements):
    """Replicate eval_short_gaps.py carving. Returns (windows, records).
    record = (gap_len_L, score_index_array(in-window), true_values)."""
    S = max(1, stride)
    windows, records = [], []
    n_skipped = {L: 0 for L in gaps}
    for pi, samp in enumerate(test):
        full_ts = samp["irg_ts"][:, 0].astype(np.float32)
        full_orig = samp["irg_ts_mask"][:, 0].astype(bool)
        if full_orig.shape[0] < t_hi:
            for L in gaps:
                n_skipped[L] += 1
            continue
        day2_ts = full_ts[t_lo:t_hi]
        day2_obs = full_orig[t_lo:t_hi]
        for L in gaps:
            rng = np.random.default_rng(1000 * pi + L)
            score_off = list(range(0, L, S))
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
                len(cands), size=min(n_placements, len(cands)), replace=False
            )
            for idx in picks:
                s = cands[int(idx)]
                win = np.where(day2_obs, day2_ts, np.nan).astype(np.float32)
                loc = s - t_lo  # gap start within the day-2 window
                win[loc : loc + L] = np.nan  # carve the contiguous gap
                score_idx = np.array([loc + j for j in score_off], dtype=int)
                windows.append(win)
                records.append((L, score_idx, full_ts[[s + j for j in score_off]]))
    return np.stack(windows, 0), records, n_skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["phys", "shortgap"])
    ap.add_argument("--method", required=True, choices=["saits", "brits"])
    ap.add_argument("--dataset", default="aireadi")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--n_participants", type=int, default=352)
    ap.add_argument("--n_masks", type=int, default=5, help="phys: masks per strategy")
    ap.add_argument(
        "--gaps", default="3,6,9,12", help="shortgap: gap lengths in 5-min samples"
    )
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument(
        "--n_placements",
        type=int,
        default=10,
        help="shortgap: gaps per (participant,L)",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--pypots_python", default=DEFAULT_PYPOTS)
    ap.add_argument(
        "--pypots_site",
        default=DEFAULT_PYPOTS_SITE,
        help="pypots site-packages injected on the subprocess PYTHONPATH",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train, val, test, meta = load_dataset_splits(
        args.dataset, prepared_dir=args.data_dir
    )
    test = test[: args.n_participants]
    cgm_std = float(meta.glucose_std_mgdl)
    t_lo, t_hi = meta.eval_window
    t0 = time.time()

    if args.mode == "phys":
        windows, records, n_skipped = collect_phys(test, t_lo, t_hi, args.n_masks)
        n_used = len(test) - n_skipped
        print(
            f"[phys] {len(windows)} windows over {n_used} participants "
            f"({n_skipped} skipped, no full day-2; {time.time()-t0:.0f}s); "
            f"fitting {args.method} (epochs={args.epochs})...",
            flush=True,
        )
        imputed = run_pypots(
            windows, args.method, args.pypots_python, args.epochs, args.pypots_site
        )
        rm = {s: [] for s in STRATS}
        for (strat, tgt_idx, true_vals), row in zip(records, imputed):
            rm[strat].append(rmse_mgdl(row[tgt_idx], true_vals, cgm_std))
        per = {s: (float(np.nanmean(rm[s])) if rm[s] else float("nan")) for s in STRATS}
        allv = [v for s in STRATS for v in rm[s] if v == v]
        avg = float(np.mean(allv)) if allv else float("nan")
        print(
            f"\n=== {args.method.upper()} physiological (AI-READI, n={len(test)}) ==="
        )
        for s in STRATS:
            print(f"  {s:10s} {per[s]:.2f}")
        print(f"  {'AVG':10s} {avg:.2f} mg/dL")
        out = {
            "method": args.method,
            "mode": "phys",
            "per_strategy": per,
            "avg": avg,
            "n_participants": n_used,
            "n_requested": len(test),
            "n_skipped_participants": n_skipped,
            "n_masks": args.n_masks,
            "n_windows": len(windows),
            "epochs": args.epochs,
            "transductive": True,
            "dataset": meta.name,
        }
    else:
        gaps = [int(x) for x in args.gaps.split(",") if x.strip()]
        windows, records, n_skipped = collect_shortgap(
            test, t_lo, t_hi, gaps, args.stride, args.n_placements
        )
        print(
            f"[shortgap] {len(windows)} windows over {len(test)} participants "
            f"({time.time()-t0:.0f}s); fitting {args.method} (epochs={args.epochs})...",
            flush=True,
        )
        imputed = run_pypots(
            windows, args.method, args.pypots_python, args.epochs, args.pypots_site
        )
        per_gap = {L: [] for L in gaps}
        for (L, score_idx, true_vals), row in zip(records, imputed):
            per_gap[L].append(rmse_mgdl(row[score_idx], true_vals, cgm_std))
        summary = {
            f"{L*5}min": (float(np.mean(per_gap[L])) if per_gap[L] else float("nan"))
            for L in gaps
        }
        counts = {f"{L*5}min": len(per_gap[L]) for L in gaps}
        vals = [v for L in gaps for v in per_gap[L]]
        mean = float(np.mean(vals)) if vals else float("nan")
        print(f"\n=== {args.method.upper()} short-gap (AI-READI, n={len(test)}) ===")
        for L in gaps:
            print(
                f"  {L*5:3d}min  RMSE={summary[f'{L*5}min']:.2f}  (n={counts[f'{L*5}min']})"
            )
        print(f"  mean     RMSE={mean:.2f}")
        out = {
            "method": args.method,
            "mode": "shortgap",
            "by_gap_min": summary,
            "mean": mean,
            "n_eval": counts,
            "n_skipped_participants": n_skipped,
            "n_participants": len(test),
            "n_placements": args.n_placements,
            "stride": int(max(1, args.stride)),
            "n_windows": len(windows),
            "epochs": args.epochs,
            "transductive": True,
            "dataset": meta.name,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Saved -> {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
