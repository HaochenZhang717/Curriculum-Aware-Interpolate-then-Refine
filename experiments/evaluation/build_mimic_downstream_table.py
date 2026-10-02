"""Build the MIMIC downstream table: reconstruction RMSE vs clinical-burden recovery."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

PCTS = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3"]
THR = [
    "mrr_tir",
    "mrr_tar180",
    "mrr_tbr70",
]  # threshold-burden metrics (not gamed by variability)
DISP = {
    "cair": "\\method\\ (uni)",
    "cair_mm": "\\textbf{\\method\\ (mm)}",
    "linear": "linear interp",
    "locf": "LOCF",
    "fourier": "Fourier",
    "knn": "$k$NN",
    "hotdeck": "hot-deck",
    "mice": "MICE",
    "missforest": "missForest",
    "mean": "mean",
    "mode": "mode",
    "mrnn": "MRNN",
    "gpvae": "GP-VAE",
}
# grouped for the Pareto story
GROUPS = [
    ("Interpolation (low RMSE, destroys burden)", ["linear", "locf", "fourier"]),
    (
        "Tabular / neural (recovers burden, high RMSE)",
        ["knn", "hotdeck", "mice", "mrnn", "gpvae"],
    ),
    ("\\method\\ (both)", ["cair", "cair_mm"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="abp", choices=["abp", "hr"])
    ap.add_argument("--mech", default="nmar", choices=["mcar", "mar", "nmar"])
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rd = args.results_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"
    )
    a = json.load(
        open(os.path.join(rd, f"toye_mimic_{args.target}_realistic_full_n200.json"))
    )["aggregate"]
    me = args.mech

    def rmse(m):
        v = [a[f"{m}|{me}|{p}"]["rmse"] for p in PCTS if f"{m}|{me}|{p}" in a]
        return float(np.mean(v)) if v else float("nan")

    def bmrr(m):
        v = [
            a[f"{m}|{me}|{p}"][k]
            for p in PCTS
            for k in THR
            if f"{m}|{me}|{p}" in a
            and a[f"{m}|{me}|{p}"].get(k) == a[f"{m}|{me}|{p}"].get(k)
        ]
        return float(np.mean(v)) if v else float("nan")

    unit = "mmHg" if args.target == "abp" else "bpm"
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        f"Method & RMSE ({unit}) $\\downarrow$ & Burden-MRR $\\uparrow$ \\\\",
        "\\midrule",
    ]
    best_rmse = min(rmse(m) for g in GROUPS for m in g[1])
    for title, ms in GROUPS:
        lines.append(f"\\multicolumn{{3}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for m in ms:
            r, b = rmse(m), bmrr(m)
            rs = f"\\textbf{{{r:.2f}}}" if abs(r - best_rmse) < 1e-6 else f"{r:.2f}"
            row = f"\\quad {DISP[m]} & {rs} & {b:+.2f} \\\\"
            lines.append(row)
        lines.append("\\midrule" if title != GROUPS[-1][0] else "\\bottomrule")
    lines.append("\\end{tabular}")
    tex = "\n".join(lines)
    print(tex)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w").write(tex + "\n")
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
