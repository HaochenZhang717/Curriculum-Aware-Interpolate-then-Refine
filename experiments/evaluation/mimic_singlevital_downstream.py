"""Single-modality clinical downstream: does imputing ONE vital's gaps change sepsis/HF/mortality prediction when the classifier sees ONLY that vital?"""

from __future__ import annotations

from utils.paths import DATA_ROOT, REPO_ROOT, CACHE_ROOT

import argparse
import glob as _glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (ROOT, os.path.join(ROOT, "baselines", "statistical")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cgm_datasets import load_dataset_splits
from utils.missingness_mechanisms import (
    create_nmar_mask,
    create_mcar_mask,
    native_gap_durations,
)
from experiments.evaluation.mimic_clinical_downstream import (
    derive_labels,
    to_native,
    complete,
)

LABELS = ["mortality", "sepsis", "hf"]


def prognostic_features(x, lo, hi):
    """Single-vital prognostic features, hypotension/tachycardia-burden focused."""
    x = np.asarray(x, np.float64).reshape(-1)
    n = len(x)
    if n == 0:
        return np.zeros(16, np.float32)
    below = x < lo
    above = x > hi
    # longest contiguous run below lo
    longest = cur = 0
    for b in below:
        cur = cur + 1 if b else 0
        longest = max(longest, cur)
    auc_below = float(np.clip(lo - x, 0, None).sum() / n)  # mean depth below lo
    auc_above = float(np.clip(x - hi, 0, None).sum() / n)
    dx = np.diff(x) if n > 1 else np.array([0.0])
    return np.array(
        [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.min(x)),
            float(np.max(x)),
            float(np.median(x)),
            float(np.percentile(x, 10)),
            float(np.percentile(x, 90)),
            float(below.mean()),
            float(above.mean()),
            float(longest) / n,
            auc_below,
            auc_above,
            float(100.0 * np.std(x) / np.mean(x)) if np.mean(x) else 0.0,  # CV
            float(np.mean(np.abs(dx))),
            float(np.max(np.abs(dx))) if len(dx) else 0.0,
            float(np.polyfit(np.arange(n), x, 1)[0]) if n > 1 else 0.0,
        ],
        np.float32,
    )


def build_clf(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=4,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=seed,
    )


def fit_reference(train_recs, labels, mean, std, lo, hi, n_seeds=5):
    from sklearn.preprocessing import StandardScaler

    X, ys = [], {k: [] for k in LABELS}
    for r in train_recs:
        sid = r.get("stay_id")
        if sid is None or int(sid) not in labels:
            continue
        tr, obs = to_native(r, mean, std)
        X.append(prognostic_features(complete(tr, obs), lo, hi))
        for k in LABELS:
            ys[k].append(int(labels[int(sid)][k]))
    X = np.vstack(X)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    models = {}
    for k in LABELS:
        y = np.asarray(ys[k])
        if len(np.unique(y)) < 2 or y.sum() < 20:
            continue
        models[k] = [build_clf(s).fit(Xs, y) for s in range(n_seeds)]
    return models, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mimic_abp")
    ap.add_argument(
        "--source_pkl", default=str(DATA_ROOT / "raw" / "mimic3_waveform_abp.pkl")
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_test", type=int, default=400)
    ap.add_argument("--n_train", type=int, default=0, help="0=all")
    ap.add_argument("--mech", default="nmar", choices=["nmar", "mcar"])
    ap.add_argument("--rate", type=float, default=0.3)
    ap.add_argument(
        "--rates",
        default=None,
        help="csv of NMAR rates to sweep, e.g. 0.05,0.1,0.15,0.2,0.25,0.3",
    )
    ap.add_argument(
        "--eval_split",
        default="test",
        choices=["test", "valtest"],
        help="valtest pools val+test (both unseen by the classifier) for power",
    )
    ap.add_argument("--methods", default="real,mean,linear,refine,refine_mm")
    ap.add_argument("--ckpt_suffix", default="_realistic")
    ap.add_argument(
        "--signal_only", action="store_true", help="just report oracle AUROC and stop"
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tr, va, te, meta = load_dataset_splits(args.dataset)
    mean, std = float(meta.glucose_mean_mgdl), float(meta.glucose_std_mgdl)
    lo, hi = (65.0, 100.0) if "hr" not in args.dataset else (50.0, 100.0)
    if args.n_train:
        tr = tr[: args.n_train]
    # eval set: test only, or val+test pooled (val is unseen by the classifier that
    # trains on the train split, so pooling is leak-free and adds statistical power).
    eval_recs = (list(te) + list(va)) if args.eval_split == "valtest" else list(te)
    if args.n_test:
        eval_recs = eval_recs[: args.n_test]
    te = eval_recs
    labels = derive_labels(args.source_pkl)
    print(f"[data] train={len(tr)} test={len(te)} lo/hi={lo}/{hi}", flush=True)
    models, sc = fit_reference(tr, labels, mean, std, lo, hi)
    print(f"[train] labels with signal: {list(models.keys())}", flush=True)

    from sklearn.metrics import roc_auc_score

    def auroc_for(feature_rows, y):
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            return None
        Xs = sc.transform(np.vstack(feature_rows))
        aus = []
        for k_models in [models[k] for k in [lab]]:
            probs = np.mean([m.predict_proba(Xs)[:, 1] for m in k_models], axis=0)
            aus.append(roc_auc_score(y, probs))
        return float(np.mean(aus))

    # oracle-signal check
    print(
        "\n[oracle signal] single-vital AUROC (native-completed real traces):",
        flush=True,
    )
    for lab in [k for k in LABELS if k in models]:
        feats, ys = [], []
        for r in te:
            sid = r.get("stay_id")
            if sid is None or int(sid) not in labels:
                continue
            trc, obs = to_native(r, mean, std)
            feats.append(prognostic_features(complete(trc, obs), lo, hi))
            ys.append(int(labels[int(sid)][lab]))
        au = auroc_for(feats, ys)
        print(
            f"  {lab}: oracle AUROC = {au:.3f} (n={len(ys)}, pos={int(np.sum(ys))})",
            flush=True,
        )
    if args.signal_only:
        return

    # full method comparison
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    refine = refine_mm = None
    ts_mods = ["hr", "resp", "spo2"]
    ckdir = os.path.join(
        ROOT, "method_checkpoints", f"refine_{args.dataset}{args.ckpt_suffix}"
    )
    if "refine" in methods:
        p = sorted(_glob.glob(os.path.join(ckdir, "uni_seed*.pt")))
        if p:
            from methods.refine import RefineImputer

            refine = RefineImputer(ckpt_paths=p, device=args.device)
        else:
            methods = [m for m in methods if m != "refine"]
    if "refine_mm" in methods:
        p = sorted(_glob.glob(os.path.join(ckdir, "mm_seed*.pt")))
        if p:
            import torch
            from methods.refine import RefineImputer

            refine_mm = RefineImputer(ckpt_paths=p, device=args.device)
            ts_mods = (
                torch.load(p[0], map_location="cpu")
                .get("config", {})
                .get("ts_mods", ts_mods)
                or ts_mods
            )
        else:
            methods = [m for m in methods if m != "refine_mm"]

    from cgm_datasets.multimodal.modality_spec import build_mod_and_ctx
    from baselines_autoregressive import pchip_impute  # noqa: F401  (ensure path)
    from simple import linear_interp

    rates = [float(x) for x in args.rates.split(",")] if args.rates else [args.rate]
    collected = {
        (lab, m, rate): ([], []) for lab in models for m in methods for rate in rates
    }
    for i, r in enumerate(te):
        sid = r.get("stay_id")
        if sid is None or int(sid) not in labels:
            continue
        trc, obs = to_native(r, mean, std)
        full = complete(trc, obs)
        sz = r["irg_ts"]
        om = r["irg_ts_mask"]
        gd = native_gap_durations(om) or [3, 6, 9, 12]
        mod = ctx = None
        if refine_mm is not None:
            mod, ctx = build_mod_and_ctx(r, ts_mods, [])
        for rate in rates:
            seed = 9000 + i * 10 + int(rate * 100)
            if args.mech == "nmar":
                _e, tm = create_nmar_mask(
                    sz,
                    om,
                    percent=rate,
                    gap_durations=gd,
                    seed=seed,
                    mean_mgdl=mean,
                    std_mgdl=std,
                    lo_mgdl=lo,
                    hi_mgdl=hi,
                )
            else:
                _e, tm = create_mcar_mask(sz, om, percent=rate, seed=seed)
            tgt = tm[:, 0].astype(bool)
            if tgt.sum() == 0:
                continue
            obsb = om[:, 0].astype(bool) & ~tgt  # what a method sees
            zmask = obsb.astype(np.float32)
            for m in methods:
                filled = full.copy()
                if m == "real":
                    pass  # oracle: keep true values in the gap
                elif m == "mean":
                    filled[tgt] = (
                        float(full[obsb].mean()) if obsb.any() else float(full.mean())
                    )
                elif m == "linear":
                    z = sz[:, 0].astype(np.float32).copy()
                    pr = np.asarray(
                        linear_interp(np.where(obsb, z, np.nan), zmask)
                    ).reshape(-1)
                    filled[tgt] = (pr * std + mean)[tgt]
                elif m in ("refine", "refine_mm"):
                    z = sz[:, 0].astype(np.float32).copy()
                    z[~obsb] = 0.0
                    imp = refine if m == "refine" else refine_mm
                    pr = imp.impute(
                        z,
                        zmask,
                        modality=(mod if m == "refine_mm" else None),
                        ctx_vec=(ctx if m == "refine_mm" else None),
                    )
                    filled[tgt] = (np.asarray(pr).reshape(-1) * std + mean)[tgt]
                feat = prognostic_features(filled, lo, hi)
                for lab in models:
                    collected[(lab, m, rate)][0].append(feat)
                    collected[(lab, m, rate)][1].append(int(labels[int(sid)][lab]))
        if i % 50 == 0:
            print(f"  eval {i}/{len(te)}", flush=True)

    results = []
    order = [
        m for m in ("real", "refine_mm", "refine", "linear", "mean") if m in methods
    ]
    for lab in models:
        print(
            f"\n[result] {lab}: single-vital AUROC by NMAR rate (n_pos={int(np.sum(collected[(lab, order[0], rates[0])][1]))})",
            flush=True,
        )
        print("  " + "rate".ljust(7) + "".join(f"{m:>11}" for m in order), flush=True)
        for rate in rates:
            row = "  " + f"{rate:<7}"
            for m in order:
                feats, ys = collected[(lab, m, rate)]
                au = auroc_for(feats, ys) if feats else None
                results.append(
                    {
                        "label": lab,
                        "method": m,
                        "mech": args.mech,
                        "rate": rate,
                        "auroc": au,
                        "n": len(ys),
                    }
                )
                row += f"{(f'{au:.3f}' if au is not None else '-'):>11}"
            print(row, flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(
            {"dataset": args.dataset, "results": results}, open(args.out, "w"), indent=2
        )
        print(f"[out] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
