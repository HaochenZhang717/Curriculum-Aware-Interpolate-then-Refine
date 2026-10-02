"""Assemble MIMIC result tables (RMSE, downstream MRR, careunit/mortality strata) from an eval_toye_mimic cell JSON. Emits markdown; LaTeX rows are trivial to derive."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

MECHS = ["mcar", "mar", "nmar"]
MRR_KEYS = ["mrr_tir", "mrr_tar180", "mrr_tbr70", "mrr_mage", "mrr_cv"]
METHOD_ORDER = [
    "refine",
    "linear",
    "locf",
    "mean",
    "mode",
    "knn",
    "hotdeck",
    "mice",
    "missforest",
    "fourier",
]


def _mean(vals):
    vals = [v for v in vals if v is not None and v == v]
    return float(np.mean(vals)) if vals else float("nan")


def _by(cells, keyfn, valfn):
    d = defaultdict(list)
    for c in cells:
        d[keyfn(c)].append(valfn(c))
    return d


def _order(methods):
    known = [m for m in METHOD_ORDER if m in methods]
    return known + sorted(m for m in methods if m not in METHOD_ORDER)


def rmse_table(cells, units):
    methods = {c["method"] for c in cells}
    agg = _by(cells, lambda c: (c["method"], c["mech"]), lambda c: c["rmse"])
    lines = [
        f"| method | "
        + " | ".join(f"{m.upper()} RMSE ({units})" for m in MECHS)
        + " |",
        "|---|" + "---|" * len(MECHS),
    ]
    for m in _order(methods):
        row = [m] + [f"{_mean(agg[(m, mech)]):.3f}" for mech in MECHS]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def mrr_table(cells):
    methods = {c["method"] for c in cells}
    lines = [
        "| method | mech | " + " | ".join(MRR_KEYS) + " |",
        "|---|---|" + "---|" * len(MRR_KEYS),
    ]
    for m in _order(methods):
        for mech in MECHS:
            sub = [c for c in cells if c["method"] == m and c["mech"] == mech]
            row = [m, mech] + [f"{_mean([c[k] for c in sub]):.3f}" for k in MRR_KEYS]
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def strata_table(cells, units):
    methods = _order({c["method"] for c in cells})
    out = ["### By care unit\n"]
    cus = sorted({str(c["careunit"]) for c in cells})
    out.append("| method | " + " | ".join(cus) + " |")
    out.append("|---|" + "---|" * len(cus))
    for m in methods:
        row = [m] + [
            f"{_mean([c['rmse'] for c in cells if c['method']==m and str(c['careunit'])==cu]):.3f}"
            for cu in cus
        ]
        out.append("| " + " | ".join(row) + " |")
    out.append("\n### By in-hospital mortality (0=survived,1=died)\n")
    out.append("| method | survived | died |")
    out.append("|---|---|---|")
    for m in methods:
        row = [m] + [
            f"{_mean([c['rmse'] for c in cells if c['method']==m and int(c['mortality'])==mt]):.3f}"
            for mt in (0, 1)
        ]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="eval_toye_mimic output JSON")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    data = json.load(open(args.cells))
    cells = data["cells"]
    ds = data["dataset"]
    units = data.get("units", "")
    os.makedirs(args.outdir, exist_ok=True)
    open(os.path.join(args.outdir, f"{ds}_rmse.md"), "w").write(
        rmse_table(cells, units)
    )
    open(os.path.join(args.outdir, f"{ds}_mrr.md"), "w").write(mrr_table(cells))
    open(os.path.join(args.outdir, f"{ds}_strata.md"), "w").write(
        strata_table(cells, units)
    )
    print(f"wrote {ds}_rmse.md, {ds}_mrr.md, {ds}_strata.md to {args.outdir}")


if __name__ == "__main__":
    main()
