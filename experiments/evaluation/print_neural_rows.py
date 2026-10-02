#!/usr/bin/env python3
"""Emit the LaTeX table rows for the SAITS/BRITS cells once the eval jobs finish. Reads results/neural_phys_{m}_n352.json and results/shortgap_{m}.json."""

import json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "results")
STR = ["meal_post", "sleep", "ascending", "dipping", "combined"]
GAP = ["15min", "30min", "45min", "60min"]


def f(path):
    return json.load(open(path)) if os.path.exists(path) else None


print("=== Table 1 (physiological) rows ===")
for m, disp in [("saits", "SAITS"), ("brits", "BRITS")]:
    d = f(os.path.join(R, f"neural_phys_{m}_n352.json"))
    if not d:
        print(f"  {disp}: (pending)")
        continue
    p = d["per_strategy"]
    cells = " & ".join(f"{p[s]:.2f}" for s in STR)
    print(
        f"    {disp}~\\citep{{{m}}}             & {cells} & {d['avg']:.2f} \\\\"
        f"   % n={d['n_participants']} epochs={d['epochs']} windows={d['n_windows']}"
    )

print("\n=== Table 2 (short-gap) rows ===")
for m, disp in [("saits", "SAITS"), ("brits", "BRITS")]:
    d = f(os.path.join(R, f"shortgap_{m}.json"))
    if not d:
        print(f"  {disp}: (pending)")
        continue
    g = d["by_gap_min"]
    cells = " & ".join(f"{g[k]:.2f}" for k in GAP)
    print(
        f"    {disp}~\\citep{{{m}}}                 & {cells} & {d['mean']:.2f} \\\\"
        f"   % epochs={d['epochs']} windows={d['n_windows']}"
    )
