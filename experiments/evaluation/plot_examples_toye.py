"""Render fig5/fig6-style method-comparison imputation galleries for the Toye (MCAR/MAR/NMAR) setups: MIMIC-III ABP, MIMIC-III HR, AI-READI CGM."""

from __future__ import annotations

from utils.paths import DATA_ROOT, REPO_ROOT, CACHE_ROOT

import argparse
import csv
import json
import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

MM = 1.0 / 25.4
COL1, COL2 = 89 * MM, 183 * MM

# Okabe-Ito colour-blind-safe palette. REFINE = blue (ours).
COLOR = {
    "refine": "#0072B2",  # blue   - ours
    "linear": "#999999",  # grey
    "knn": "#E69F00",  # orange
    "gpvae": "#CC79A7",  # reddish purple
    "mrnn": "#009E73",  # bluish green
    "mice": "#D55E00",  # vermillion
    "missforest": "#56B4E9",  # sky blue
    "locf": "#F0E442",  # yellow
    "mean": "#661100",
    "mode": "#332288",
    "hotdeck": "#AA4499",
    "fourier": "#117733",
    # classical interpolators (cross-dataset short-gap set)
    "akima": "#E69F00",  # orange
    "pchip": "#009E73",  # green
    "spline": "#CC79A7",  # reddish purple
    "savgol": "#D55E00",  # vermillion
}
LABEL = {
    "refine": "REFINE (ours)",
    "linear": "linear",
    "knn": "k-NN",
    "gpvae": "GP-VAE",
    "mrnn": "MRNN",
    "mice": "MICE",
    "missforest": "missForest",
    "locf": "LOCF",
    "mean": "mean-fill",
    "mode": "mode-fill",
    "hotdeck": "hot-deck",
    "fourier": "Fourier",
    "akima": "akima",
    "pchip": "PCHIP",
    "spline": "cubic spline",
    "savgol": "Sav-Gol",
}
BALANCED = ["linear", "knn", "gpvae", "mrnn", "refine"]  # refine drawn last
FULL13 = [
    "linear",
    "locf",
    "mean",
    "mode",
    "knn",
    "hotdeck",
    "mice",
    "missforest",
    "fourier",
    "gpvae",
    "mrnn",
    "refine",
]
MECHS = ["mcar", "mar", "nmar"]
MECH_TITLE = {"mcar": "MCAR", "mar": "MAR", "nmar": "NMAR"}
TRUTH_C, OBS_C = "#222222", "#000000"
DATASET_TITLE = {
    "mimic_abp": "MIMIC-III ABP",
    "mimic_hr": "MIMIC-III HR",
    "aireadi": "AI-READI CGM",
}


def set_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.0,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


class Setup:
    def __init__(self, base):
        self.base = base
        self.d = np.load(base + ".npz")
        self.meta = json.load(open(base + "_meta.json"))
        self.std = float(self.meta["glucose_std_mgdl"])
        self.mean = float(self.meta["glucose_mean_mgdl"])
        self.L = int(self.meta["L"])
        self.dt = float(self.meta["sampling_minutes"])
        self.units = self.meta["units"]
        self.methods = self.meta["methods"]
        self.windows = self.meta["windows"]
        # x axis: hours if the window spans >= 4 h, else minutes
        span_min = self.L * self.dt
        if span_min >= 240:
            self.x = np.arange(self.L) * self.dt / 60.0
            self.xlabel = "time (h)"
        else:
            self.x = np.arange(self.L) * self.dt
            self.xlabel = "time (min)"
        self.hw = (self.x[1] - self.x[0]) * 0.5

    def mgdl(self, key, w):
        return self.d[key][w] * self.std + self.mean


def panel_label(ax, s, x=-0.10, y=1.06):
    ax.text(
        x,
        y,
        s,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def pick(windows, mech, pct, ranks, min_target=8):
    """Return window dicts for the given (mech, pct) at the requested difficulty
    ranks (fractional positions 0..1 by REFINE RMSE, ascending)."""
    pool = [
        w
        for w in windows
        if w["mech"] == mech
        and abs(w["pct"] - pct) < 1e-6
        and w["n_target"] >= min_target
        and "refine" in w["rmse"]
    ]
    if not pool:
        pool = [w for w in windows if w["mech"] == mech and abs(w["pct"] - pct) < 1e-6]
    if not pool:
        return []
    pool = sorted(pool, key=lambda w: w["rmse"]["refine"])
    out, used = [], set()
    for fr in ranks:
        i = min(len(pool) - 1, max(0, int(round(fr * (len(pool) - 1)))))
        while i in used and i < len(pool) - 1:
            i += 1
        used.add(i)
        out.append(pool[i])
    return out


def plot_panel(ax, S, w, methods, title=None, emphasize="refine"):
    idx = w["idx"]
    x, hw = S.x, S.hw
    truth = S.mgdl("truth", idx)
    obs = S.d["obs"][idx].astype(bool)
    tgt = S.d["tgt"][idx].astype(bool)
    orig = obs | tgt
    spans = w["gap_spans"]
    # Block-gap mechanisms (few contiguous spans) zoom to the gap so the method
    # curves fill the panel; scattered MCAR (many spans) stays full-window.
    block = len(spans) <= 3
    if block and spans:
        g0 = min(s for s, _ in spans)
        g1 = max(e for _, e in spans)
        margin = max(8, (g1 - g0))
        lo = max(0, g0 - margin)
        hi = min(S.L - 1, g1 + margin)
    else:
        lo, hi = 0, S.L - 1
    win = np.zeros(S.L, bool)
    win[lo : hi + 1] = True

    # shade masked spans (single points -> thin bars; blocks -> continuous)
    for s, e in spans:
        ax.axvspan(x[s] - hw, x[e] + hw, color="#cccccc", alpha=0.45, lw=0, zorder=0)
    # ground truth where a real value exists, broken elsewhere
    ax.plot(x, np.where(orig, truth, np.nan), color=TRUTH_C, lw=0.7, zorder=2)
    # observed input dots (kept context)
    om = obs & win
    ax.plot(
        x[om],
        truth[om],
        ls="none",
        marker="o",
        ms=1.4,
        mfc=OBS_C,
        mec="none",
        alpha=0.55,
        zorder=3,
    )
    # method predictions at masked points; emphasized method drawn last
    order = [m for m in methods if m != emphasize] + (
        [emphasize] if emphasize in methods else []
    )
    # y-limits are anchored on the *meaningful* signal (truth + observed context
    # + the emphasized method) so degenerate baselines (e.g. GP-VAE/MRNN that
    # collapse to a near-constant fill) clip at the panel edge instead of
    # compressing the REFINE-vs-truth detail.
    core = [truth[orig & win], truth[om]]
    for m in order:
        if f"pred_{m}" not in S.d.files:
            continue
        pred = S.mgdl(f"pred_{m}", idx)
        is_emph = m == emphasize
        c = COLOR.get(m, "#777777")
        lw = 1.4 if is_emph else 0.9
        ms = 3.2 if is_emph else 2.0
        z = 9 if is_emph else 6
        for s, e in spans:
            if e < lo or s > hi:
                continue
            xs = x[s : e + 1]
            ys = pred[s : e + 1]
            if e > s:  # interpolation across gap
                ax.plot(
                    xs, ys, color=c, lw=lw, alpha=0.95, zorder=z, solid_capstyle="round"
                )
            mk = tgt[s : e + 1]  # markers only at held-out truth pts
            ax.plot(
                xs[mk],
                ys[mk],
                ls="none",
                marker="o",
                ms=ms,
                mfc=c,
                mec="white",
                mew=0.25 if is_emph else 0.0,
                alpha=0.95,
                zorder=z,
            )
            if is_emph:
                core.append(ys)
    ax.set_xlim(x[lo], x[hi])
    allv = np.concatenate([v for v in core if len(v)])
    if allv.size:
        ymin, ymax = float(np.nanmin(allv)), float(np.nanmax(allv))
        rng = max(ymax - ymin, 4.0)
        ax.set_ylim(ymin - 0.14 * rng, ymax + 0.14 * rng)
    if title:
        ax.set_title(title, fontsize=6, pad=2)
    ax.tick_params(labelsize=5.2)
    ax.locator_params(axis="x", nbins=5)
    ax.locator_params(axis="y", nbins=5)


def legend_handles(methods):
    h = [
        Line2D([], [], color=TRUTH_C, lw=0.7, label="ground truth"),
        Line2D(
            [],
            [],
            color=OBS_C,
            marker="o",
            ls="none",
            ms=2,
            alpha=0.5,
            label="observed (input)",
        ),
        Patch(color="#cccccc", alpha=0.6, label="masked / held-out"),
    ]
    for m in methods:
        h.append(
            Line2D(
                [],
                [],
                color=COLOR.get(m, "#777777"),
                marker="o",
                ls="-" if m == "refine" else "-",
                lw=1.3 if m == "refine" else 0.8,
                ms=3 if m == "refine" else 1.8,
                label=LABEL.get(m, m),
            )
        )
    return h


def save(fig, name, outdir):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02
        )
    plt.close(fig)
    print("wrote", name)


def rmse_title(w, methods):
    r = w["rmse"]
    rr = r.get("refine", float("nan"))
    base = {m: r[m] for m in methods if m != "refine" and m in r}
    if base:
        bm = min(base, key=base.get)
        return f"RMSE  REFINE {rr:.1f} · best base {LABEL.get(bm, bm)} {base[bm]:.1f}"
    return f"RMSE {rr:.1f}"


def fig_main(S, methods, outdir):
    methods = [m for m in methods if m in S.methods]
    rates = S.meta["rates"]
    nrow, ncol = len(MECHS), len(rates)
    fig, axes = plt.subplots(nrow, ncol, figsize=(COL2, 40 * MM * nrow), squeeze=False)
    letters = iter("abcdefghijklmnop")
    rows_tab = []
    for r, mech in enumerate(MECHS):
        for c, pct in enumerate(rates):
            ax = axes[r][c]
            sel = pick(S.windows, mech, pct, [0.5])
            if not sel:
                ax.axis("off")
                continue
            w = sel[0]
            plot_panel(
                ax,
                S,
                w,
                methods,
                title=f"{MECH_TITLE[mech]} · {int(round(pct*100))}%   "
                + rmse_title(w, methods),
            )
            panel_label(ax, next(letters))
            if c == 0:
                ax.set_ylabel(f"{dtitle(S)}\n({S.units})", fontsize=5.6)
            if r == nrow - 1:
                ax.set_xlabel(S.xlabel, fontsize=5.8)
            rows_tab.append(("main", mech, pct, w))
    fig.legend(
        handles=legend_handles(methods),
        ncol=len(methods) + 3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=6,
    )
    fig.suptitle(
        f"{dtitle(S)} — REFINE vs baselines under real-world "
        f"missingness (Toye protocol)",
        fontsize=7.5,
        y=1.005,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.99), h_pad=1.2, w_pad=0.9)
    save(fig, f"{tag(S)}_main", outdir)
    return rows_tab


def fig_mech_gallery(S, mech, methods, outdir, ncol=4):
    methods = [m for m in methods if m in S.methods]
    rates = S.meta["rates"]
    ranks = [i / (ncol - 1) for i in range(ncol)]  # easy..hard
    fig, axes = plt.subplots(
        len(rates), ncol, figsize=(COL2, 34 * MM * len(rates)), squeeze=False
    )
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    rows_tab = []
    for r, pct in enumerate(rates):
        sel = pick(S.windows, mech, pct, ranks)
        for c in range(ncol):
            ax = axes[r][c]
            if c >= len(sel):
                ax.axis("off")
                continue
            w = sel[c]
            tags = ["easiest", "", "", "hardest"]
            tt = f"{int(round(pct*100))}% · REFINE {w['rmse'].get('refine', float('nan')):.1f}"
            plot_panel(ax, S, w, methods, title=tt)
            panel_label(ax, next(letters))
            if c == 0:
                ax.set_ylabel(
                    f"{int(round(pct*100))}% missing\n({S.units})", fontsize=5.6
                )
            if r == len(rates) - 1:
                ax.set_xlabel(S.xlabel, fontsize=5.8)
            rows_tab.append((f"{mech}_gallery", mech, pct, w))
    fig.legend(
        handles=legend_handles(methods),
        ncol=len(methods) + 3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=6,
    )
    fig.suptitle(
        f"{dtitle(S)} — {MECH_TITLE[mech]} imputation examples "
        f"(easy→hard by REFINE RMSE)",
        fontsize=7.5,
        y=1.005,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.99), h_pad=1.2, w_pad=0.9)
    save(fig, f"{tag(S)}_{mech}_gallery", outdir)
    return rows_tab


def fig_full13(S, outdir, n=3):
    methods = [m for m in FULL13 if m in S.methods]
    # pick block-gap windows (mar/nmar) at the highest rate, median difficulty
    pct = max(S.meta["rates"])
    pool = []
    for mech in ("nmar", "mar"):
        pool += pick(S.windows, mech, pct, [0.35, 0.6])
    pool = pool[:n] if pool else pick(S.windows, "mcar", pct, [0.5])[:n]
    if not pool:
        return []
    fig, axes = plt.subplots(1, len(pool), figsize=(COL2, 52 * MM), squeeze=False)
    letters = iter("abcdefg")
    for i, w in enumerate(pool):
        ax = axes[0][i]
        plot_panel(
            ax,
            S,
            w,
            methods,
            title=f"{MECH_TITLE[w['mech']]} · {int(round(w['pct']*100))}%",
        )
        panel_label(ax, next(letters), x=-0.08)
        ax.set_xlabel(S.xlabel, fontsize=5.8)
        if i == 0:
            ax.set_ylabel(f"{dtitle(S)} ({S.units})", fontsize=6)
    fig.legend(
        handles=legend_handles(methods),
        ncol=6,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        fontsize=5.6,
    )
    fig.suptitle(
        f"{dtitle(S)} — all 13 methods on structured gaps", fontsize=7.5, y=1.02
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.98), w_pad=1.2)
    save(fig, f"{tag(S)}_full13", outdir)
    return [("full13", w["mech"], w["pct"], w) for w in pool]


def tag(S):
    return S.meta["dataset"].replace("/", "_")


def dtitle(S):
    return DATASET_TITLE.get(S.meta["dataset"], S.meta["dataset"])


def write_tables(S, tab_rows, outdir):
    methods = [m for m in FULL13 if m in S.methods]
    md = os.path.join(outdir, f"{tag(S)}_panel_rmse.md")
    csvp = os.path.join(outdir, f"{tag(S)}_panel_rmse.csv")
    with open(csvp, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(
            ["figure", "mech", "pct", "pid", "n_target"]
            + [f"{m}_rmse" for m in methods]
            + ["best"]
        )
        for fign, mech, pct, w in tab_rows:
            r = w["rmse"]
            best = min((m for m in methods if m in r), key=lambda m: r[m])
            wcsv.writerow(
                [fign, mech, pct, w["pid"], w["n_target"]]
                + [f"{r.get(m, float('nan')):.2f}" for m in methods]
                + [best]
            )
    # aggregate mean in-gap RMSE per method per (mech,rate) over ALL windows
    with open(md, "w") as f:
        f.write(f"# {dtitle(S)} — in-gap RMSE ({S.units})\n\n")
        f.write(
            "Mean over all captured windows, per mechanism × missingness "
            "rate. Lowest per row in **bold**.\n\n"
        )
        f.write(
            "| mech | rate | " + " | ".join(LABEL.get(m, m) for m in methods) + " |\n"
        )
        f.write("|" + "|".join(["---"] * (len(methods) + 2)) + "|\n")
        for mech in MECHS:
            for pct in S.meta["rates"]:
                ws = [
                    w
                    for w in S.windows
                    if w["mech"] == mech and abs(w["pct"] - pct) < 1e-6
                ]
                if not ws:
                    continue
                means = {
                    m: float(np.mean([w["rmse"][m] for w in ws if m in w["rmse"]]))
                    for m in methods
                }
                bm = min(means, key=means.get)
                cells = [MECH_TITLE[mech], f"{int(round(pct*100))}%"]
                for m in methods:
                    v = f"{means[m]:.1f}"
                    cells.append(f"**{v}**" if m == bm else v)
                f.write("| " + " | ".join(cells) + " |\n")
    print("wrote", md, "and", csvp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bases",
        nargs="+",
        required=True,
        help="capture base paths (without .npz), one per setup",
    )
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
    set_style()
    for base in args.bases:
        S = Setup(base)
        print(
            f"\n=== {S.meta['dataset']} : {len(S.windows)} windows, "
            f"methods={S.methods} ==="
        )
        tab = []
        tab += fig_main(S, BALANCED, outdir)
        for mech in MECHS:
            tab += fig_mech_gallery(S, mech, BALANCED, outdir)
        tab += fig_full13(S, outdir)
        write_tables(S, tab, outdir)
    print("\nall figures ->", outdir)


if __name__ == "__main__":
    main()
