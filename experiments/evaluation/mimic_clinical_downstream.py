"""Clinical downstream recovery for MIMIC-III vital imputation."""

from __future__ import annotations

from utils.paths import DATA_ROOT, REPO_ROOT, CACHE_ROOT

import argparse
import glob as _glob
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (ROOT, os.path.join(ROOT, "baselines", "statistical")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.missingness_mechanisms import (  # noqa: E402
    create_mcar_mask,
    create_nmar_mask,
    native_gap_durations,
)
from cgm_datasets import load_dataset_splits  # noqa: E402

LABELS = ("mortality", "sepsis", "hf")
MECHS = ("mcar", "nmar")
RATES = (0.2, 0.3)

# ICD-9 prefixes (dots stripped) for sepsis and heart failure.
_SEPSIS_PREFIXES = (
    "99591",
    "99592",
    "78552",
    "0380",
    "0381",
    "0382",
    "0383",
    "0384",
    "0388",
    "0389",
)
_HF_PREFIX = "428"


# Labels


def derive_labels(source_pkl):
    """Map stay_id -> {"mortality", "sepsis", "hf"} from the source pickle.

    sepsis / hf are set from the stay's ICD-9 diagnoses (code is element [1] of
    each diagnosis tuple); mortality comes from labels["in_hospital_mortality"].
    """
    with open(source_pkl, "rb") as f:
        data = pickle.load(f)
    out = {}
    for stay in data["stays"].values():
        sid = int(stay["stay_id"])
        mortality = int(stay["labels"]["in_hospital_mortality"])
        sepsis = 0
        hf = 0
        for diag in stay.get("diagnoses") or []:
            code = str(diag[1]).replace(".", "")
            if code.startswith(_SEPSIS_PREFIXES):
                sepsis = 1
            if code.startswith(_HF_PREFIX):
                hf = 1
        out[sid] = {"mortality": mortality, "sepsis": sepsis, "hf": hf}
    return out


# Trace helpers


def to_native(rec, mean, std):
    """Return (trace_native (T,), obs_mask (T,)) in native units."""
    trace = rec["irg_ts"][:, 0].astype(np.float32) * std + mean
    obs = rec["irg_ts_mask"][:, 0].astype(np.float32)
    return trace, obs


def complete(trace, obs):
    """Linear-interpolate over natively missing points (obs==0).

    Edge points are extended by nearest observed value (np.interp default). The
    result is the "native-completed" reference trace shared by all conditions.
    """
    trace = np.asarray(trace, dtype=np.float32).reshape(-1)
    obs = np.asarray(obs).reshape(-1).astype(bool)
    idx = np.arange(len(trace))
    if not obs.any():
        return np.zeros(len(trace), dtype=np.float32)
    return np.interp(idx, idx[obs], trace[obs]).astype(np.float32)


def trace_features(x, thr_lo=65.0, thr_hi=100.0):
    """Fixed clinical features from a native-unit (T,) trace.

    14 features: mean, std, min, max, median, IQR, frac<lo, frac>hi, MAGE,
    CV(%), first, last, linear slope, threshold-crossing count.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = len(x)
    if n == 0:
        return np.zeros(14, dtype=np.float32)
    mean = float(np.mean(x))
    std = float(np.std(x))
    p25, p75 = np.percentile(x, [25, 75])
    iqr = float(p75 - p25)
    frac_lo = float(np.mean(x < thr_lo))
    frac_hi = float(np.mean(x > thr_hi))
    # MAGE: mean absolute excursion between successive turning points that
    # exceeds 1 SD of the trace.
    mage = 0.0
    if std > 0 and n >= 3:
        d = np.diff(x)
        # turning points where the first difference changes sign
        tp = [0]
        for i in range(1, len(d)):
            if d[i - 1] != 0 and d[i] != 0 and np.sign(d[i - 1]) != np.sign(d[i]):
                tp.append(i)
        tp.append(n - 1)
        excursions = np.abs(np.diff(x[tp]))
        big = excursions[excursions > std]
        mage = float(np.mean(big)) if big.size else 0.0
    cv = float(100.0 * std / mean) if mean != 0 else 0.0
    first = float(x[0])
    last = float(x[-1])
    slope = float(np.polyfit(np.arange(n), x, 1)[0]) if n >= 2 else 0.0
    # threshold crossings: how often the trace crosses the low threshold.
    below = (x < thr_lo).astype(np.int8)
    crossings = int(np.sum(np.abs(np.diff(below))))
    return np.array(
        [
            mean,
            std,
            float(np.min(x)),
            float(np.max(x)),
            float(np.median(x)),
            iqr,
            frac_lo,
            frac_hi,
            mage,
            cv,
            first,
            last,
            slope,
            float(crossings),
        ],
        dtype=np.float32,
    )


def context_features(rec, context_slots):
    """Fixed features from the co-recorded (non-target) vitals, held REAL across every
    fill method. A single vital over one 24h window is near chance for stay-level labels
    (sepsis/HF/mortality); the co-recorded vitals make the label predictable while the
    target-vital imputation remains the only thing that varies between methods.

    5 features per vital (mean, std, min, max, slope) on the native-completed z-scored
    context trace. A missing slot contributes zeros.
    """
    feats = []
    for slot in context_slots:
        vk, pk = f"mm_{slot}", f"mm_{slot}_p"
        if vk in rec and pk in rec:
            v = np.asarray(rec[vk], np.float64).reshape(-1)
            p = (np.asarray(rec[pk], np.float64).reshape(-1) > 0).astype(np.float32)
            vv = complete(v, p)
            slope = (
                float(np.polyfit(np.arange(len(vv)), vv, 1)[0]) if len(vv) >= 2 else 0.0
            )
            feats.extend(
                [
                    float(np.mean(vv)),
                    float(np.std(vv)),
                    float(np.min(vv)),
                    float(np.max(vv)),
                    slope,
                ]
            )
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    return np.asarray(feats, dtype=np.float32)


def _full_feat(target_native, rec, context_slots, thr_lo, thr_hi):
    """Target-vital features (vary with the fill) concatenated with fixed context-vital
    features. When context_slots is empty, reduces to single-vital features."""
    tf = trace_features(target_native, thr_lo, thr_hi)
    if not context_slots:
        return tf
    return np.concatenate([tf, context_features(rec, context_slots)])


# Reference classifiers


def train_reference(
    train_recs, labels, mean, std, thr_lo, thr_hi, n_seeds=5, context_slots=None
):
    """Train per-label RandomForest ensembles on native-completed train traces.

    Returns {label: [RandomForestClassifier, ...]} for labels that have both
    classes and at least 20 positives; other labels are dropped.
    """
    from sklearn.ensemble import RandomForestClassifier

    X = []
    y_by_label = {lab: [] for lab in LABELS}
    for rec in train_recs:
        sid = rec.get("stay_id")
        if sid is None or int(sid) not in labels:
            continue
        trace, obs = to_native(rec, mean, std)
        full = complete(trace, obs)
        X.append(_full_feat(full, rec, context_slots, thr_lo, thr_hi))
        lab = labels[int(sid)]
        for name in LABELS:
            y_by_label[name].append(int(lab[name]))
    if not X:
        return {}
    X = np.vstack(X)

    models = {}
    for name in LABELS:
        y = np.asarray(y_by_label[name], dtype=int)
        if len(np.unique(y)) < 2 or int(y.sum()) < 20:
            print(
                f"  [skip] label '{name}': classes={np.unique(y).tolist()} "
                f"positives={int(y.sum())}",
                flush=True,
            )
            continue
        clfs = []
        for seed in range(n_seeds):
            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                random_state=seed,
                class_weight="balanced",
                n_jobs=1,
            )
            clf.fit(X, y)
            clfs.append(clf)
        models[name] = clfs
        print(
            f"  [train] label '{name}': n={len(y)} pos={int(y.sum())} "
            f"({100.0 * y.mean():.1f}%), {n_seeds} seeds",
            flush=True,
        )
    return models


def _predict_prob(clfs, feat):
    """Mean P(y=1) over a classifier ensemble for a single feature vector."""
    feat = feat.reshape(1, -1)
    probs = []
    for clf in clfs:
        # class index of the positive label
        pos = int(np.where(clf.classes_ == 1)[0][0]) if 1 in clf.classes_ else -1
        p = clf.predict_proba(feat)[0]
        probs.append(p[pos] if pos >= 0 else 0.0)
    return float(np.mean(probs))


# Method fills (all operate on the artificial gap only)


def _fill_methods_native(
    rec,
    full_native,
    obs_native,
    target_mask,
    mean,
    std,
    methods,
    refine=None,
    refine_mm=None,
    mod=None,
    ctx=None,
):
    """Return {method: filled_native_trace (T,)} for the requested methods.

    `full_native` is the native-completed reference; `target_mask` marks the
    artificially deleted positions. Each method only changes those positions.
    """
    from simple import linear_interp  # z-space linear interp baseline

    T = len(full_native)
    tgt = target_mask.astype(bool)
    # seen mask in native/eval space: observed AND not artificially deleted.
    seen = obs_native.astype(bool) & (~tgt)
    z = rec["irg_ts"][:, 0].astype(np.float32)
    out = {}

    for m in methods:
        if m == "real":
            filled = full_native.copy()  # oracle: keeps true native values
        elif m == "mean":
            filled = full_native.copy()
            obs_mean = (
                float(full_native[seen].mean())
                if seen.any()
                else float(full_native.mean())
            )
            filled[tgt] = obs_mean
        elif m == "linear":
            # linear interp in z-space over the seen (post-deletion) points.
            zpred = linear_interp(z, seen.astype(np.float32))[:, 0] * std + mean
            filled = full_native.copy()
            filled[tgt] = zpred[tgt]
        elif m == "knn":
            from baselines.toye.toye_baselines import impute_knn

            zpred = impute_knn(z, seen.astype(np.float32))
            zpred = np.asarray(zpred, dtype=np.float32).reshape(-1) * std + mean
            filled = full_native.copy()
            filled[tgt] = zpred[tgt]
        elif m in ("refine", "refine_mm"):
            imp = refine if m == "refine" else refine_mm
            if imp is None:
                continue
            fm = z.copy()
            fm[~seen] = 0.0
            if m == "refine_mm":
                zpred = imp.impute(
                    fm, seen.astype(np.float32), modality=mod, ctx_vec=ctx
                )
            else:
                zpred = imp.impute(fm, seen.astype(np.float32))
            zpred = np.asarray(zpred, dtype=np.float32).reshape(-1) * std + mean
            filled = full_native.copy()
            filled[tgt] = zpred[tgt]
        else:
            continue
        out[m] = filled.astype(np.float32)
    return out


# Evaluation


def evaluate(
    test_recs,
    labels,
    models,
    mean,
    std,
    thr_lo,
    thr_hi,
    methods,
    refine=None,
    refine_mm=None,
    ts_mods=None,
    mechs=MECHS,
    rates=RATES,
    seed_base=7000,
    context_slots=None,
):
    """Run the imputation-isolation eval over the test split.

    Returns a list of result dicts, one per (label, method, mech, rate).
    """
    from sklearn.metrics import roc_auc_score

    ts_mods = ts_mods or ["hr", "resp", "spo2"]
    need_mm = "refine_mm" in methods and refine_mm is not None
    if need_mm:
        from cgm_datasets.multimodal.modality_spec import build_mod_and_ctx

    # collected[(label, method, mech, rate)] = ([probs], [ys])
    collected = {}
    for li, rec in enumerate(test_recs):
        sid = rec.get("stay_id")
        if sid is None or int(sid) not in labels:
            continue
        rec_labels = labels[int(sid)]
        trace, obs = to_native(rec, mean, std)
        full_native = complete(trace, obs)
        series_z = rec["irg_ts"]  # (L,1) z-space
        obs_mask_z = rec["irg_ts_mask"]  # (L,1)
        gap_durs = native_gap_durations(obs_mask_z) or [3, 6, 9, 12]
        mod, ctx = (None, None)
        if need_mm:
            mod, ctx = build_mod_and_ctx(rec, ts_mods, [])

        for mech in mechs:
            for rate in rates:
                seed = seed_base + li
                if mech == "mcar":
                    _em, tm = create_mcar_mask(
                        series_z, obs_mask_z, percent=rate, seed=seed
                    )
                else:
                    _em, tm = create_nmar_mask(
                        series_z,
                        obs_mask_z,
                        percent=rate,
                        gap_durations=gap_durs,
                        seed=seed,
                        mean_mgdl=mean,
                        std_mgdl=std,
                        lo_mgdl=thr_lo,
                        hi_mgdl=thr_hi,
                    )
                target_mask = tm[:, 0].astype(np.float32)
                if target_mask.sum() == 0:
                    continue
                fills = _fill_methods_native(
                    rec,
                    full_native,
                    obs,
                    target_mask,
                    mean,
                    std,
                    methods,
                    refine=refine,
                    refine_mm=refine_mm,
                    mod=mod,
                    ctx=ctx,
                )
                for method, filled in fills.items():
                    feat = _full_feat(filled, rec, context_slots, thr_lo, thr_hi)
                    for name, clfs in models.items():
                        p = _predict_prob(clfs, feat)
                        key = (name, method, mech, rate)
                        collected.setdefault(key, ([], []))
                        collected[key][0].append(p)
                        collected[key][1].append(int(rec_labels[name]))
        if li % 50 == 0:
            print(f"  test window {li}/{len(test_recs)}", flush=True)

    results = []
    for (name, method, mech, rate), (probs, ys) in collected.items():
        ys = np.asarray(ys, dtype=int)
        probs = np.asarray(probs, dtype=float)
        auroc = None
        if len(np.unique(ys)) >= 2:
            auroc = float(roc_auc_score(ys, probs))
        results.append(
            {
                "label": name,
                "method": method,
                "mech": mech,
                "rate": rate,
                "auroc": auroc,
                "n": int(len(ys)),
            }
        )
    return results


def _print_table(results, mech="nmar", rate=0.3):
    order = ["real", "refine_mm", "refine", "linear", "knn", "mean"]
    print(f"\n=== AUROC at {mech} rate {rate} ===")
    header = "label      " + "".join(f"{m:>11}" for m in order)
    print(header)
    for name in LABELS:
        row = f"{name:<11}"
        for m in order:
            hit = [
                r
                for r in results
                if r["label"] == name
                and r["method"] == m
                and r["mech"] == mech
                and r["rate"] == rate
            ]
            if hit and hit[0]["auroc"] is not None:
                row += f"{hit[0]['auroc']:>11.3f}"
            else:
                row += f"{'-':>11}"
        print(row)


# Driver


def _thresholds_for(dataset):
    """Native-unit clinical thresholds (lo, hi). ABP mmHg vs HR bpm."""
    if "hr" in dataset:
        return 50.0, 100.0
    return 65.0, 100.0  # ABP default


def run(args):
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    train, _val, test, meta = load_dataset_splits(args.dataset)
    mean = float(meta.glucose_mean_mgdl)
    std = float(meta.glucose_std_mgdl)
    thr_lo, thr_hi = _thresholds_for(args.dataset)
    test = test[: args.n_test]
    # Context vitals: the co-recorded (non-target) vitals whose features are held REAL
    # across every fill method. Auto-detected from the mm_* keys unless given explicitly.
    context_vitals = getattr(args, "context_vitals", "auto")
    if context_vitals == "none":
        context_slots = []
    elif context_vitals == "auto":
        probe = test[0] if test else (train[0] if train else {})
        context_slots = [
            s
            for s in ("hr", "resp", "spo2", "abp")
            if f"mm_{s}" in probe and f"mm_{s}_p" in probe
        ]
    else:
        context_slots = [s.strip() for s in context_vitals.split(",") if s.strip()]
    print(
        f"[data] {meta.name}: train={len(train)} test(eval)={len(test)} "
        f"mean={mean:.2f} std={std:.2f} thr=({thr_lo},{thr_hi}) "
        f"context_vitals={context_slots}",
        flush=True,
    )

    print("[labels] deriving from source pickle ...", flush=True)
    labels = derive_labels(args.source_pkl)
    print(f"[labels] {len(labels)} stays labeled", flush=True)

    print(
        "[train] fitting reference classifiers on native-completed traces ...",
        flush=True,
    )
    models = train_reference(
        train,
        labels,
        mean,
        std,
        thr_lo,
        thr_hi,
        n_seeds=args.n_seeds,
        context_slots=context_slots,
    )
    if not models:
        print("[warn] no usable labels; aborting", flush=True)

    # REFINE ensembles (GPU). Guard: skip a method if its checkpoints are absent.
    # Ensembles live under method_checkpoints/refine_<dataset>/ as uni_seed*.pt
    # (unimodal -> "refine") and mm_seed*.pt (multimodal -> "refine_mm").
    refine = None
    refine_mm = None
    ts_mods = ["hr", "resp", "spo2"]
    device = args.device
    ckpt_dir = os.path.join(
        ROOT,
        "method_checkpoints",
        f"refine_{args.dataset}{getattr(args, 'ckpt_suffix', '')}",
    )
    if "refine" in methods:
        rpaths = sorted(_glob.glob(os.path.join(ckpt_dir, "uni_seed*.pt")))
        if rpaths:
            from methods.refine import RefineImputer

            refine = RefineImputer(ckpt_paths=rpaths, device=device)
        else:
            print(
                f"[skip] method 'refine': no uni_seed*.pt under {ckpt_dir}", flush=True
            )
            methods = [m for m in methods if m != "refine"]
    if "refine_mm" in methods:
        mpaths = sorted(_glob.glob(os.path.join(ckpt_dir, "mm_seed*.pt")))
        if mpaths:
            from methods.refine import RefineImputer

            refine_mm = RefineImputer(ckpt_paths=mpaths, device=device)
            # read ts_mods off the first checkpoint config when available
            try:
                import torch

                cfg = torch.load(mpaths[0], map_location="cpu").get("config", {})
                ts_mods = cfg.get("ts_mods", ts_mods) or ts_mods
            except Exception:
                pass
        else:
            print(
                f"[skip] method 'refine_mm': no mm_seed*.pt under {ckpt_dir}",
                flush=True,
            )
            methods = [m for m in methods if m != "refine_mm"]

    print(
        f"[eval] methods={methods} mechs={list(MECHS)} rates={list(RATES)}", flush=True
    )
    results = evaluate(
        test,
        labels,
        models,
        mean,
        std,
        thr_lo,
        thr_hi,
        methods,
        refine=refine,
        refine_mm=refine_mm,
        ts_mods=ts_mods,
        context_slots=context_slots,
    )

    out = {
        "dataset": meta.name,
        "labels": list(models.keys()),
        "results": results,
        "bounds_note": "real=oracle upper, mean=lower",
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[out] wrote {args.out} ({len(results)} rows)", flush=True)

    _print_table(results, mech="nmar", rate=0.3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mimic_abp")
    ap.add_argument(
        "--source_pkl", default=str(DATA_ROOT / "raw" / "mimic3_waveform_abp.pkl")
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_test", type=int, default=400)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--methods", default="real,mean,linear,knn,refine,refine_mm")
    ap.add_argument(
        "--context_vitals",
        default="auto",
        help="'auto' (co-recorded mm_* vitals), 'none' (single-vital), or a csv",
    )
    ap.add_argument(
        "--ckpt_suffix",
        default="",
        help="REFINE checkpoint dir suffix, e.g. '_realistic' for mask-aligned",
    )
    ap.add_argument("--out", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
