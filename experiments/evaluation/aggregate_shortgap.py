#!/usr/bin/env python3
"""Aggregate short-gap RMSE JSONs for one target into a dual-unit markdown table."""

from __future__ import annotations
import argparse, glob, json, os

GAPS = ["15min", "30min", "45min", "60min"]
# display order: baselines then the proposed method
ORDER = [
    "linear",
    "forward_fill",
    "ewma",
    "savgol",
    "spline",
    "akima",
    "ar",
    "pchip",
    "cair",
    "cair_mm",
]
# marks a method as ours in the table
OURS = {"cair": " **(ours, unimodal)**", "cair_mm": " **(ours, multimodal)**"}


def load_rows(results_dir, target):
    rows = {}
    for p in glob.glob(os.path.join(results_dir, f"{target}_shortgap_*.json")):
        d = json.load(open(p))
        method = os.path.basename(p)[len(target) + len("_shortgap_") : -len(".json")]
        rows[method] = d["by_gap_min"]
    return rows


def fmt_table(rows, native_std, title, znorm):
    lines = [
        f"### {title}",
        "",
        "| Method | " + " | ".join(GAPS) + " |",
        "|" + "---|" * (len(GAPS) + 1),
    ]
    ordered = [m for m in ORDER if m in rows] + [m for m in rows if m not in ORDER]
    for m in ordered:
        cells = []
        for g in GAPS:
            v = rows[m].get(g)
            if v is None:
                cells.append("N/A")
            elif znorm:
                cells.append(f"{v / native_std:.3f}")
            else:
                cells.append(f"{v:.2f}")
        tag = OURS.get(m, "")
        lines.append(f"| {m}{tag} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--native_std", type=float, required=True)
    ap.add_argument("--unit", default=None, help="native unit label, e.g. bpm")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    unit = args.unit or {"hr": "bpm", "resp": "breaths/min"}.get(args.target, "native")
    rows = load_rows(args.results_dir, args.target)
    if not rows:
        raise SystemExit(
            f"no result JSONs found for target={args.target} in {args.results_dir}"
        )
    parts = [
        f"# Short-gap imputation RMSE: {args.target.upper()}",
        "",
        f"Native std = {args.native_std:.4f} {unit}. "
        f"Native table is RMSE in {unit}; z-norm table divides by the native std.",
        "",
        fmt_table(rows, args.native_std, f"Native units ({unit})", znorm=False),
        "",
        fmt_table(rows, args.native_std, "Z-normalized (dimensionless)", znorm=True),
        "",
    ]
    report = "\n".join(parts)
    with open(args.out, "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
