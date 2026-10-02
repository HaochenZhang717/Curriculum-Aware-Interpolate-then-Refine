#!/usr/bin/env python3
"""Aggregate the MULTI-SEED (5-seed ensemble) architectural ablation."""

from __future__ import annotations
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ABL = os.path.join(ROOT, "results", "ablation")
GAPS = ["15min", "30min", "45min", "60min"]

ARCH = [
    ("full", "control"),
    ("no_aux", "-- stage-1 interpolation supervision (diff vs SAITS)"),
    ("no_resid", "-- residual target (refiner predicts residual by construction)"),
    ("attn_interp", "-- bi-GRU interpolator -> attention block"),
]


def load_strat(cell):
    fp = os.path.join(ABL, f"strat_{cell}_ens5.json")
    if not os.path.exists(fp):
        return None
    d = json.load(open(fp))
    return {
        "per": d.get("per_strategy", {}),
        "avg": d.get("avg", float("nan")),
        "nref": d.get("n_refinements", 2),
    }


def load_short(cell):
    fp = os.path.join(ABL, f"shortgap_{cell}_ens5.json")
    if not os.path.exists(fp):
        return None
    d = json.load(open(fp))
    bg = d.get("by_gap_min", {})
    vals = [bg.get(g, float("nan")) for g in GAPS]
    m = sum(vals) / 4 if all(v == v for v in vals) else float("nan")
    return {"by_gap": vals, "mean": m}


def main():
    full = load_strat("full")
    fa = full["avg"] if full else float("nan")

    print(
        "\n=== CAIR architectural ablation, 5-SEED ENSEMBLE (AI-READI, 5-strategy protocol, N=50) ==="
    )
    print(
        "  Primary metric: avg RMSE mg/dL over meal/sleep/ascending/dipping/combined."
    )

    # collect strategy keys from full if present
    skeys = list(full["per"].keys()) if full and full["per"] else []
    hdr = "  {:<14}".format("cell")
    for k in skeys:
        hdr += f" {k[:8]:>8}"
    hdr += f" {'AVG':>7} {'dVsFull':>8}   tests"
    print(hdr)
    print("  " + "-" * (len(hdr) + 4))
    for cell, tests in ARCH:
        r = load_strat(cell)
        if not r:
            print(f"  {cell:<14} (pending)")
            continue
        line = f"  {cell:<14}"
        for k in skeys:
            line += f" {r['per'].get(k, float('nan')):8.2f}"
        dv = "  --  " if cell == "full" else f"{r['avg']-fa:+8.2f}"
        line += f" {r['avg']:7.2f} {dv:>8}   {tests}"
        print(line)

    print("\n  -- refinement-pass sweep on full ensemble (5-strategy; control=2) --")
    for nr in [0, 1, 2, 3]:
        if nr == 2:
            r = load_strat("full")
        else:
            fp = os.path.join(ABL, f"strat_passes_{nr}_ens5.json")
            r = None
            if os.path.exists(fp):
                d = json.load(open(fp))
                r = {"avg": d.get("avg", float("nan"))}
        tag = f"passes={nr}" + (" (=full)" if nr == 2 else "")
        if r:
            dv = "  --  " if nr == 2 else f"{r['avg']-fa:+.2f}"
            print(f"  {tag:<16} avg={r['avg']:.2f}  dVsFull={dv}")
        else:
            print(f"  {tag:<16} (pending)")

    print("\n  -- secondary: short-gap ensemble (mean over 15/30/45/60 min) --")
    for cell, _ in ARCH:
        r = load_short(cell)
        if r:
            dvs = ""
            fs = load_short("full")
            if fs and cell != "full":
                dvs = f"  dVsFull={r['mean']-fs['mean']:+.2f}"
            print(
                f"  {cell:<14} "
                + " ".join(f"{v:5.2f}" for v in r["by_gap"])
                + f"  mean={r['mean']:5.2f}{dvs}"
            )
        else:
            print(f"  {cell:<14} (pending)")
    print()


if __name__ == "__main__":
    main()
