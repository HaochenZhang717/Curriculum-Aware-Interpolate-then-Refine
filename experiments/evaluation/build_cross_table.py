#!/usr/bin/env python3
"""Assemble the cross-dataset zero-shot short-gap tables from per-cell JSONs."""

from __future__ import annotations
import json, os, sys, glob, math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results", "cross_dataset")
GAPS = ["15min", "30min", "45min", "60min"]

# external eval targets (order = table order)
EXTERNAL = [
    ("ohio", "OhioT1DM", "T1D", "5-min"),
    ("hupa_ucm", "HUPA-UCM", "T1D", "5-min"),
    ("shanghai", "Shanghai T1DM+T2DM", "T1D + T2DM", "15-min"),
    ("mimic_abp", "MIMIC-III ABP", "ICU", "5-min"),
    ("mimic_hr", "MIMIC-III HR", "ICU", "5-min"),
]
# method key -> display label ; row order (CAIR first, then interpolators, AR, smoothers, naive)
METHODS = [
    ("cair", "CAIR (zero-shot)"),
    ("akima", "akima"),
    ("pchip", "PCHIP"),
    ("linear", "linear"),
    ("spline", "cubic spline"),
    ("ar", "AR (bidir)"),
    ("savgol", "Savitzky-Golay"),
    ("ewma", "EWMA"),
    ("forward_fill", "forward-fill"),
]
# external datasets also get the fine-tuned (per-dataset K-fold CV) row as the headline "our method"
EXT_METHODS = [("cair_ft", "CAIR (fine-tuned, CV)")] + METHODS
# same rows for the in-domain reference block, only the CAIR label changes (no fine-tuned/zero-shot split)
RLABELS = [("cair", "CAIR (in-domain)")] + METHODS[1:]


def load_cv_finetuned(dataset):
    """Pool the per-dataset K-fold fine-tuned evals into one cell.

    Each fold tested a disjoint set of held-out subjects; pooling is exact via
    n-weighted RMSE^2 (RMSE^2 = mean squared error, so combine MSE weighted by count).
    """
    fs = sorted(
        glob.glob(os.path.join(RES, "cv", f"shortgap_{dataset}_fold*_finetuned.json"))
    )
    if not fs:
        return None
    num = {g: 0.0 for g in GAPS}
    den = {g: 0 for g in GAPS}
    nfolds = 0
    for f in fs:
        d = json.load(open(f))
        bg = d.get("by_gap_min", {})
        ne = d.get("n_eval", {})
        if any(
            bg.get(g) is None or bg.get(g) != bg.get(g) or not ne.get(g) for g in GAPS
        ):
            continue
        nfolds += 1
        for g in GAPS:
            num[g] += ne[g] * (bg[g] ** 2)
            den[g] += ne[g]
    if nfolds == 0 or any(den[g] == 0 for g in GAPS):
        return None
    bg = {g: math.sqrt(num[g] / den[g]) for g in GAPS}
    return {
        "by_gap": bg,
        "mean": sum(bg.values()) / len(GAPS),
        "n_eval": den[GAPS[0]],
        "stride": 1,
        "n_participants": None,
        "n_folds": nfolds,
    }


def load_cell(dataset, method):
    if method == "cair_ft":
        return load_cv_finetuned(dataset)
    p = os.path.join(RES, f"shortgap_{dataset}_{method}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    bg = d.get("by_gap_min", {})
    vals = [bg.get(g) for g in GAPS]
    if any(v is None or v != v for v in vals):  # missing / NaN
        return None
    mean = sum(vals) / len(vals)
    n = d.get("n_eval", {}).get("15min")
    return {
        "by_gap": {g: bg[g] for g in GAPS},
        "mean": mean,
        "n_eval": n,
        "stride": d.get("stride", 1),
        "n_participants": d.get("n_participants"),
    }


def fmt_table(rows, methods):
    """rows: {method_key: cell}. Bold the column-wise minimum (lower RMSE better)."""
    cols = GAPS + ["mean"]
    # column minima across available methods
    mins = {}
    for c in cols:
        vv = [
            (
                rows[m]["mean" if c == "mean" else None]
                if False
                else (rows[m]["mean"] if c == "mean" else rows[m]["by_gap"][c])
            )
            for m, _ in methods
            if rows.get(m)
        ]
        mins[c] = min(vv) if vv else None
    head = "| method | 15 min | 30 min | 45 min | 60 min | **mean** |"
    sep = "|--------|:------:|:------:|:------:|:------:|:--------:|"
    lines = [head, sep]
    for mkey, mlabel in methods:
        cell = rows.get(mkey)
        if cell is None:
            lines.append(f"| {mlabel} | - | - | - | - | - |")
            continue
        vals = [cell["by_gap"][g] for g in GAPS] + [cell["mean"]]
        cells = []
        for c, v in zip(cols, vals):
            s = f"{v:.2f}"
            if mins[c] is not None and abs(v - mins[c]) < 1e-9:
                s = f"**{s}**"
            cells.append(s)
        lines.append(f"| {mlabel} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    combined = {}
    md = []
    md.append("# Cross-dataset short-gap RMSE (mg/dL)\n")
    md.append(
        "Two CAIR rows per external cohort: **fine-tuned** = the AI-READI model "
        "fine-tuned on that dataset, tested on held-out subjects via K-fold "
        "cross-validation (Ohio leave-one-out=6 folds; HUPA/Shanghai 5-fold); "
        "**zero-shot** = the published AI-READI ensemble applied with no retraining. "
        "Classical baselines (akima / PCHIP / linear / ...) are training-free. Gap is a "
        "single contiguous dropout carved into the day-2 window at originally-observed "
        "positions; RMSE over target positions only, lower is better, **bold** = best in "
        "column.\n"
    )

    for key, label, cohort, native in EXTERNAL:
        rows = {m: load_cell(key, m) for m, _ in EXT_METHODS}
        combined[key] = {
            "cohort": cohort,
            "native_cadence": native,
            "cells": {m: (rows[m] if rows[m] else None) for m, _ in EXT_METHODS},
        }
        ft = rows.get("cair_ft")
        nfolds = ft.get("n_folds") if ft else None
        note = ""
        if native == "15-min":
            note = (
                " Shanghai is native 15-min CGM, so gaps are carved/scored at the native "
                "cadence (stride=3); not directly comparable to the 5-min cohorts."
            )
        foldnote = f" Fine-tuned row pooled over {nfolds} CV folds." if nfolds else ""
        md.append(f"\n## {label}  ({cohort}, native {native})\n")
        md.append(fmt_table(rows, EXT_METHODS))
        md.append(
            f"\n*Zero-shot & baseline rows use all subjects; the fine-tuned row tests only "
            f"each fold's held-out subjects.{foldnote}{note}*\n"
        )

    # AI-READI in-domain reference (SAME published ensemble; all cells on the test split)
    ref = {m: load_cell("aireadi", m) for m, _ in METHODS}
    ref = {m: c for m, c in ref.items() if c}
    if ref:
        combined["aireadi_reference"] = {
            "cohort": "in-domain (trained here)",
            "cells": {m: ref.get(m) for m, _ in METHODS},
        }
        md.append("\n## AI-READI (in-domain reference)\n")
        md.append(fmt_table(ref, RLABELS))
        md.append(
            "\n*Same published CAIR ensemble, evaluated in-domain on the AI-READI "
            "**test** split (no pooling; the model trained on AI-READI train). Shows the "
            "reference CAIR-vs-baselines gap on data from its own distribution.*\n"
        )

    out_md = os.path.join(ROOT, "docs", "cross_dataset_shortgap.md")
    out_json = os.path.join(ROOT, "results", "cross_dataset_shortgap.json")
    open(out_md, "w").write("\n".join(md) + "\n")
    json.dump(combined, open(out_json, "w"), indent=2)
    print("\n".join(md))
    print(f"\nSaved -> {out_md}\nSaved -> {out_json}")


if __name__ == "__main__":
    main()
