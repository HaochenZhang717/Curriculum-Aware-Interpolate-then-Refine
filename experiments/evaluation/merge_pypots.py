#!/usr/bin/env python3
"""Merge PyPOTS-only Toye results into an existing benchmark JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.evaluation.eval_toye_benchmark import MECHS, PERCENTS, aggregate_cells

VALUE_KEYS = [
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--add", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.base, "r", encoding="utf-8") as f:
        base = json.load(f)
    with open(args.add, "r", encoding="utf-8") as f:
        add = json.load(f)

    base["cells"] = list(base.get("cells", [])) + list(add.get("cells", []))
    agg = aggregate_cells(base["cells"], value_keys=VALUE_KEYS)
    base["aggregate"] = {f"{m}|{me}|{p}": v for (m, me, p), v in agg.items()}
    base["skipped"] = None

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=2)

    for method in ("mrnn", "gpvae"):
        for mech in MECHS:
            rmse_by_pct = []
            for pct in PERCENTS:
                cell = agg.get((method, mech, pct))
                if cell is not None and "rmse" in cell:
                    rmse_by_pct.append(f"{pct:.2f}={cell['rmse']:.6f}")
            if rmse_by_pct:
                print(f"{method} {mech} rmse {' '.join(rmse_by_pct)}")


if __name__ == "__main__":
    main()
