#!/usr/bin/env python3
"""Aggregate the architectural ablation on the REALISTIC-MISSINGNESS (Toye MCAR/MAR/NMAR) AI-READI CGM protocol."""

from __future__ import annotations
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ABL = os.path.join(ROOT, "results", "ablation")
MECHS = ["mcar", "mar", "nmar"]
ARCH = [
    ("full", "control"),
    ("no_aux", "-- stage-1 interpolation supervision (diff vs SAITS)"),
    ("no_resid", "-- residual target (refiner predicts residual by construction)"),
    ("attn_interp", "-- bi-GRU interpolator -> attention block"),
]


def cell_means(path, method="refine"):
    """Return (overall_mean, {mech: mean}) of RMSE for `method` in one toye json."""
    if not os.path.exists(path):
        return None
    agg = json.load(open(path)).get("aggregate", {})
    by_mech = {m: [] for m in MECHS}
    allv = []
    for k, v in agg.items():
        meth, mech, pct = k.split("|")
        if meth != method or "rmse" not in v:
            continue
        by_mech.setdefault(mech, []).append(v["rmse"])
        allv.append(v["rmse"])
    if not allv:
        return None
    mm = {
        m: (sum(by_mech[m]) / len(by_mech[m]) if by_mech[m] else float("nan"))
        for m in MECHS
    }
    return sum(allv) / len(allv), mm


def main():
    fullp = os.path.join(ABL, "toye_full.json")
    full = cell_means(fullp, "refine")
    fa = full[0] if full else float("nan")

    print(
        "\n=== REFINE architectural ablation | REALISTIC missingness (Toye MCAR/MAR/NMAR), AI-READI CGM ==="
    )
    print(
        "  RMSE mg/dL, mean over missingness rate 0.05-0.30. 5-seed ensemble per cell."
    )
    print(
        "  Published-ensemble ref: refine 12.16 vs best baseline linear 14.73 (n=352).\n"
    )
    print(
        f"  {'cell':<14} {'MCAR':>7} {'MAR':>7} {'NMAR':>7} {'ALL':>7} {'dVsFull':>8}   tests"
    )
    print("  " + "-" * 92)
    for cell, tests in ARCH:
        r = cell_means(os.path.join(ABL, f"toye_{cell}.json"), "refine")
        if not r:
            print(f"  {cell:<14} (pending)")
            continue
        ov, mm = r
        dv = "  --  " if cell == "full" else f"{ov-fa:+8.2f}"
        print(
            f"  {cell:<14} {mm['mcar']:7.2f} {mm['mar']:7.2f} {mm['nmar']:7.2f} {ov:7.2f} {dv:>8}   {tests}"
        )

    # baseline floor from the full cell's own recomputed baselines
    print("\n  -- classical / other baselines (from full cell's run; same masks) --")
    for b in ["linear", "locf", "knn", "mice", "missforest", "hotdeck", "mean"]:
        r = cell_means(fullp, b)
        if r:
            ov, mm = r
            print(
                f"  {b:<14} {mm['mcar']:7.2f} {mm['mar']:7.2f} {mm['nmar']:7.2f} {ov:7.2f}"
            )
    print()


if __name__ == "__main__":
    main()
