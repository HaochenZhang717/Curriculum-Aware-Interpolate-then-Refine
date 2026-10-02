#!/usr/bin/env python3
"""Aggregate results/ladder_*.json into one markdown table (the modality ladder)."""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results")

RUNGS = [
    ("r0_control", "0 control (matched)"),
    ("r1_hr", "1 +HR"),
    ("r2_steps", "2 +steps/cal"),
    ("r3_sleep", "3 +sleep"),
    ("r4_resp", "4 +resp/stress"),
    ("r5_env", "5 +environment"),
    ("r6_clinical", "6 +clinical"),
    ("r7_ecg", "7 +ECG"),
    ("r8_retinal", "8 +retinal"),
]
STRATS = ["meal_post", "sleep", "ascending", "dipping", "combined"]


def _load(prefix, kind):
    p = os.path.join(RES, f"ladder_{prefix}_{kind}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    rows = []
    for prefix, label in RUNGS:
        s5 = _load(prefix, "5strat")
        sg = _load(prefix, "shortgap")
        if s5 is None and sg is None:
            continue
        per = (s5 or {}).get("per_strategy", {})
        avg = (s5 or {}).get("avg", float("nan"))
        sgm = (sg or {}).get("by_gap_min", {})
        short_mean = (
            (sum(v for v in sgm.values() if v == v) / len(sgm)) if sgm else float("nan")
        )
        rows.append((label, per, avg, sgm, short_mean))

    def f(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) and x == x else "-"

    lines = []
    lines.append("# Multimodal REFINE: modality ladder (AI-READI test)\n")
    lines.append(
        "5-member ensemble per rung. Lead metric: 5-strategy avg + sleep/ascending. "
        "Published unimodal ensemble anchor: 14.61 (5-strat).\n"
    )
    lines.append(
        "| rung | meal_post | sleep | ascending | dipping | combined | **5-strat avg** | short-range mean |"
    )
    lines.append(
        "|------|:---------:|:-----:|:---------:|:-------:|:--------:|:---------------:|:----------------:|"
    )
    base_avg = None
    for label, per, avg, sgm, short_mean in rows:
        if base_avg is None:
            base_avg = avg
        delta = ""
        if (
            isinstance(avg, float)
            and isinstance(base_avg, float)
            and avg == avg
            and base_avg == base_avg
        ):
            d = avg - base_avg
            delta = f" ({d:+.2f})" if abs(d) > 1e-9 else ""
        lines.append(
            f"| {label} | {f(per.get('meal_post'))} | {f(per.get('sleep'))} | "
            f"{f(per.get('ascending'))} | {f(per.get('dipping'))} | {f(per.get('combined'))} | "
            f"**{f(avg)}**{delta} | {f(short_mean)} |"
        )
    if sgm:
        lines.append(
            "\nShort-range by gap length (min) for the last rung loaded: "
            + ", ".join(f"{k}={f(v)}" for k, v in sgm.items())
        )
    out = "\n".join(lines) + "\n"
    dst = os.path.join(ROOT, "docs", "multimodal_ladder_results.md")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w").write(out)
    print(out)
    print(f"saved -> {dst}")


if __name__ == "__main__":
    main()
