"""Merge the MIMIC full-benchmark result JSONs (baselines + PyPOTS + CAIR variants) into one per-target file with a unified aggregate, and print the 12-method comparison."""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

VKS = [
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
PCTS = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3"]
ORDER = [
    "linear",
    "locf",
    "mean",
    "mode",
    "knn",
    "hotdeck",
    "mice",
    "missforest",
    "fourier",
    "mrnn",
    "gpvae",
    "cair",
    "cair_mm",
]


def _load_cells(path, relabel=None):
    if not path or not os.path.exists(path):
        return []
    cells = json.load(open(path))["cells"]
    if relabel:
        for c in cells:
            c["method"] = relabel
    return cells


def _agg(cells):
    b = defaultdict(lambda: defaultdict(list))
    for c in cells:
        k = (c["method"], c["mech"], c["pct"])
        for vk in VKS:
            if vk in c and c[vk] == c[vk]:
                b[k][vk].append(c[vk])
    return {
        f"{m}|{me}|{p}": {vk: float(np.mean(v)) for vk, v in d.items()}
        for (m, me, p), d in b.items()
    }


def _mean_rmse(agg, m, mech):
    v = [agg[f"{m}|{mech}|{p}"]["rmse"] for p in PCTS if f"{m}|{mech}|{p}" in agg]
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["abp", "hr"])
    ap.add_argument("--suffix", default="_realistic", help="CAIR ckpt suffix used")
    ap.add_argument("--n", default="200")
    ap.add_argument("--results_dir", default=None)
    args = ap.parse_args()
    rd = args.results_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"
    )
    rd = os.path.abspath(rd)
    t, suf, n = args.target, args.suffix, args.n

    base = _load_cells(os.path.join(rd, f"toye_mimic_{t}_baselines_n{n}.json"))
    pypots = _load_cells(os.path.join(rd, f"toye_mimic_{t}_pypots_n{n}.json"))
    runi = _load_cells(
        os.path.join(rd, f"toye_mimic_{t}_cair{suf}_uni_n{n}.json"), "cair"
    )
    rmm = _load_cells(
        os.path.join(rd, f"toye_mimic_{t}_cair{suf}_mm_n{n}.json"), "cair_mm"
    )
    cells = base + pypots + runi + rmm
    if not cells:
        raise SystemExit(f"no cells found for target {t} under {rd}")
    units = ""
    for p in (f"toye_mimic_{t}_baselines_n{n}.json",):
        fp = os.path.join(rd, p)
        if os.path.exists(fp):
            units = json.load(open(fp)).get("units", "")
    agg = _agg(cells)
    out = {
        "dataset": f"mimic_{t}",
        "units": units,
        "suffix": suf,
        "methods": sorted({c["method"] for c in cells}),
        "cells": cells,
        "aggregate": agg,
    }
    outp = os.path.join(rd, f"toye_mimic_{t}{suf}_full_n{n}.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"wrote {outp}: {len(cells)} cells, methods={out['methods']}")

    methods = [m for m in ORDER if m in {c["method"] for c in cells}]
    print(
        f"\n=== MIMIC-{t.upper()} ({units}) full benchmark, RMSE mean/6 rates (CAIR={suf}) ==="
    )
    print(f"  {'method':12s} {'MCAR':>7s} {'MAR':>7s} {'NMAR':>7s}")
    for m in methods:
        print(
            f"  {m:12s} {_mean_rmse(agg,m,'mcar'):7.2f} "
            f"{_mean_rmse(agg,m,'mar'):7.2f} {_mean_rmse(agg,m,'nmar'):7.2f}"
        )


if __name__ == "__main__":
    main()
