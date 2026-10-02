"""Render fig5/fig6-style short-gap imputation galleries for the cross-dataset zero-shot setups (Ohio / HUPA-UCM / Shanghai): REFINE (zero-shot) vs classical interpolators (akima, PCHIP, linear, cubic spline, Savitzky-Golay).."""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

import plot_examples_toye as T  # shared style, palette, Setup, plot_panel, ...

METHODS = ["akima", "pchip", "linear", "spline", "savgol", "refine"]  # refine last


def dtitle(S):
    return S.meta.get("display_name", S.meta["dataset"])


def fig_dataset(S, outdir, ncol=4):
    methods = [m for m in METHODS if m in S.methods]
    gaps = S.meta["gaps_min"]
    ranks = [i / (ncol - 1) for i in range(ncol)]
    fig, axes = T.plt.subplots(
        len(gaps), ncol, figsize=(T.COL2, 32 * T.MM * len(gaps)), squeeze=False
    )
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    tab_rows = []
    for r, g in enumerate(gaps):
        sel = T.pick(S.windows, f"gap{g}min", float(g), ranks, min_target=1)
        for c in range(ncol):
            ax = axes[r][c]
            if c >= len(sel):
                ax.axis("off")
                continue
            w = sel[c]
            tt = f"{g} min · REFINE {w['rmse'].get('refine', float('nan')):.1f}"
            T.plot_panel(ax, S, w, methods, title=tt)
            T.panel_label(ax, next(letters))
            if c == 0:
                ax.set_ylabel(f"{g}-min gap\n(mg dL$^{{-1}}$)", fontsize=5.6)
            if r == len(gaps) - 1:
                ax.set_xlabel(S.xlabel, fontsize=5.8)
            tab_rows.append((g, w))
    fig.legend(
        handles=T.legend_handles(methods),
        ncol=len(methods) + 3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=6,
    )
    fig.suptitle(
        f"{dtitle(S)} — zero-shot REFINE vs classical interpolators "
        f"(short-gap, easy→hard by REFINE RMSE)",
        fontsize=7.5,
        y=1.005,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.99), h_pad=1.2, w_pad=0.9)
    T.save(fig, f"shortgap_{S.meta['dataset']}", outdir)
    return methods, tab_rows


def write_table(S, methods, tab_rows, outdir):
    csvp = os.path.join(outdir, f"shortgap_{S.meta['dataset']}_panel_rmse.csv")
    with open(csvp, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(
            ["gap_min", "pid", "n_target"] + [f"{m}_rmse" for m in methods] + ["best"]
        )
        for g, w in tab_rows:
            r = w["rmse"]
            best = min((m for m in methods if m in r), key=lambda m: r[m])
            wcsv.writerow(
                [g, w["pid"], w["n_target"]]
                + [f"{r.get(m, float('nan')):.2f}" for m in methods]
                + [best]
            )
    md = os.path.join(outdir, f"shortgap_{S.meta['dataset']}_gap_rmse.md")
    with open(md, "w") as f:
        f.write(f"# {dtitle(S)} — short-gap in-gap RMSE (mg/dL), zero-shot\n\n")
        f.write(
            "Mean over all captured placements, per gap length. Lowest per "
            "row in **bold**.\n\n"
        )
        f.write("| gap | " + " | ".join(T.LABEL.get(m, m) for m in methods) + " |\n")
        f.write("|" + "|".join(["---"] * (len(methods) + 1)) + "|\n")
        for g in S.meta["gaps_min"]:
            ws = [w for w in S.windows if w["gap_len_min"] == g]
            if not ws:
                continue
            means = {
                m: float(np.mean([w["rmse"][m] for w in ws if m in w["rmse"]]))
                for m in methods
            }
            bm = min(means, key=means.get)
            cells = [f"{g} min"]
            for m in methods:
                v = f"{means[m]:.1f}"
                cells.append(f"**{v}**" if m == bm else v)
            f.write("| " + " | ".join(cells) + " |\n")
    print("wrote", csvp, "and", md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", required=True)
    ap.add_argument(
        "--outdir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "paper",
            "figs",
            "examples",
        ),
    )
    args = ap.parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    T.set_style()
    for base in args.bases:
        S = T.Setup(base)
        print(f"\n=== {S.meta['dataset']} : {len(S.windows)} windows ===")
        methods, tab = fig_dataset(S, outdir)
        write_table(S, methods, tab, outdir)
    print("\nall short-gap figures ->", outdir)


if __name__ == "__main__":
    main()
