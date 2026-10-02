#!/usr/bin/env python3
"""Aggregate the architectural-ablation short-gap JSONs into one table."""

from __future__ import annotations
import glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ABL = os.path.join(ROOT, "results", "ablation")
GAPS = ["15min", "30min", "45min", "60min"]


def load(path):
    with open(path) as f:
        d = json.load(f)
    bg = d.get("by_gap_min", {})
    vals = [bg.get(g, float("nan")) for g in GAPS]
    mean = sum(vals) / len(vals) if all(v == v for v in vals) else float("nan")
    return {
        "by_gap": vals,
        "mean": mean,
        "n_ref": d.get("n_refinements"),
        "n_eval": d.get("n_eval", {}),
    }


def row(name, tests, r, base_mean):
    cells = "  ".join(f"{v:6.2f}" for v in r["by_gap"])
    dv = r["mean"] - base_mean
    dstr = "  --  " if abs(dv) < 1e-9 else f"{dv:+6.2f}"
    print(f"  {name:<14} {cells}   {r['mean']:6.2f}   {dstr}   {tests}")


def main():
    def p(cell):
        fp = os.path.join(ABL, f"shortgap_{cell}.json")
        return load(fp) if os.path.exists(fp) else None

    full = p("full")
    base_mean = full["mean"] if full else float("nan")

    print(
        "\n=== CAIR architectural ablation (AI-READI, N=50 short-gap headline protocol) ==="
    )
    print(
        f"  RMSE mg/dL by gap length.  full = control (residual, aux=0.7, gru interp, 8-layer, infer 2 passes).\n"
    )
    print(
        f"  {'cell':<14} {'15min':>6} {'30min':>6} {'45min':>6} {'60min':>6}   {'mean':>6}   {'dvsFull':>7}   tests"
    )
    print("  " + "-" * 96)

    ARCH = [
        ("full", "control"),
        ("no_aux", "-- stage-1 interpolation supervision (diff vs SAITS)"),
        ("no_resid", "-- residual target (refiner predicts residual by construction)"),
        ("attn_interp", "-- bi-GRU interpolator -> attention block"),
    ]
    for cell, tests in ARCH:
        r = p(cell)
        if r:
            row(cell, tests, r, base_mean)
        else:
            print(f"  {cell:<14} (pending)")

    print("\n  -- refinement-pass sweep (eval-only on full ckpt; control = 2) --")
    for nr in [0, 1, 2, 3]:
        r = p("full") if nr == 2 else p(f"passes_{nr}")
        tag = "passes=%d%s" % (nr, " (=full)" if nr == 2 else "")
        if r:
            row(tag, "inference refinement passes", r, base_mean)
        else:
            print(f"  {tag:<14} (pending)")

    print("\n  -- classical baselines (floor) --")
    for m in ["akima", "pchip", "linear"]:
        r = p(f"baseline_{m}")
        if r:
            row(m, "classical interpolation", r, base_mean)
        else:
            print(f"  {m:<14} (pending)")

    print()


if __name__ == "__main__":
    main()
