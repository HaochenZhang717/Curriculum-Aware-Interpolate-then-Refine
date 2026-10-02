"""Plot CGM reconstructions across conditioning modalities."""

from utils.paths import DATA_ROOT, REPO_ROOT, CACHE_ROOT
import os, sys, json, csv, argparse, subprocess, string
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CGM_MEAN, CGM_STD = 132.05, 42.33
T_LO, T_HI = 288, 576  # day-2 window
DT_MIN = 5.0  # 5 min per step
L = T_HI - T_LO
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(HERE, "_imputation_examples_cache.npz")
FIGDIR = os.path.join(REPO, "figs")

ZOOM_MARGIN = 3  # samples of context either side of the gap (~15 min)

# ladder prefix -> short human label naming the conditioning modality.
# The number in parentheses is the cumulative #conditioning modalities.
RUNG_LABEL = {
    "r0_control": "0  control",
    "r1_hr": "+HR",
    "r2_steps": "+steps/cal",
    "r3_sleep": "+sleep",
    "r4_resp": "+resp/stress",
    "r5_env": "+environment",
    "r6_clinical": "+clinical",
    "r7_ecg": "+ECG",
    "r8_retinal": "+retinal",
}

# Discrete, colourblind-safe Okabe-Ito qualitative palette. The control rung is
# drawn as a thick neutral dark grey reference line; each added-modality rung
# gets a distinct vivid colour. (Okabe & Ito 2008; safe for the common forms of
# colour-vision deficiency.)
RUNG_COLOR = {
    "r0_control": "#3a3a3a",  # neutral dark grey (reference)
    "r1_hr": "#E69F00",  # orange
    "r2_steps": "#56B4E9",  # sky blue
    "r3_sleep": "#009E73",  # bluish green
    "r4_resp": "#F0E442",  # yellow
    "r5_env": "#0072B2",  # blue
    "r6_clinical": "#D55E00",  # vermillion
    "r7_ecg": "#CC79A7",  # reddish purple
    "r8_retinal": "#000000",  # black (10 modalities; full conditioning)
}
CONTROL_PREFIX = "r0_control"

STRAT_ROW = ["sleep", "ascending", "meal_post", "dipping", "combined"]
STRAT_TITLE = {
    "sleep": "overnight / sleep",
    "ascending": "ascending / rise",
    "meal_post": "post-meal",
    "dipping": "dipping / fall",
    "combined": "mixed / combined",
}


def z2mgdl(z):
    return np.asarray(z) * CGM_STD + CGM_MEAN


def set_nature_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10.0,
            "axes.titlesize": 12.0,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 12.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.solid_capstyle": "round",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_cache():
    if not os.path.exists(CACHE):
        raise FileNotFoundError(
            f"cache not found: {CACHE}. Run with --rebuild (or run "
            "_select_imputation_examples.py first)."
        )
    d = np.load(CACHE, allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    rungs = [tuple(r) for r in json.loads(str(d["rungs_json"]))]  # (prefix, idx, cc)
    return d, meta, rungs


def order_panels(meta):
    """Return a list of 25 panel-meta dicts in row-major order
    (row = strategy in STRAT_ROW order, col = participants as cached)."""
    by_strat = {s: [] for s in STRAT_ROW}
    for m in meta:
        by_strat[m["strategy"]].append(m)
    ordered = []
    for s in STRAT_ROW:
        ordered.extend(by_strat[s])
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="regenerate the inference cache before plotting",
    )
    args = ap.parse_args()

    if args.rebuild or not os.path.exists(CACHE):
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{REPO}:{REPO}/baselines/statistical"
        subprocess.run(
            [sys.executable, os.path.join(HERE, "_select_imputation_examples.py")],
            cwd=REPO,
            env=env,
            check=True,
        )
        # ensure the retinal rung is present in the cache too
        subprocess.run(
            [sys.executable, os.path.join(HERE, "_extend_cache_r8.py")],
            cwd=REPO,
            env=env,
            check=True,
        )

    d, meta, rungs = load_cache()
    set_nature_style()
    os.makedirs(FIGDIR, exist_ok=True)

    prefixes = [p for p, _, _ in rungs]
    cond_count = {p: cc for p, _, cc in rungs}
    rung_color = {p: RUNG_COLOR.get(p, "#777777") for p in prefixes}

    ordered = order_panels(meta)
    assert len(ordered) == 25, f"expected 25 panels, got {len(ordered)}"

    # 5x5 grid; large canvas (~235 mm wide) so nothing is cramped.
    fig, axes = plt.subplots(5, 5, figsize=(235 / 25.4, 250 / 25.4))
    panel_letters = list(string.ascii_lowercase)[:25]

    x_min = np.arange(L) * DT_MIN  # minutes within day-2 window

    for k, m in enumerate(ordered):
        r, c = divmod(k, 5)
        ax = axes[r, c]
        i = m["idx"]
        obs = d[f"p{i}_obs"].astype(bool)
        truth = z2mgdl(d[f"p{i}_truth_z"])
        g0, g1 = int(m["gap_start"]), int(m["gap_end"])
        span = slice(g0, g1 + 1)

        # TIGHT zoom window: gap +/- small margin, clamped to the day-2 window
        lo = max(0, g0 - ZOOM_MARGIN)
        hi = min(L - 1, g1 + ZOOM_MARGIN)
        # x in minutes RELATIVE to gap start (readable, centred on the gap)
        x_rel = x_min - x_min[g0]
        win = (np.arange(L) >= lo) & (np.arange(L) <= hi)

        # shade the gap
        ax.axvspan(x_rel[g0], x_rel[g1], color="0.91", zorder=0, lw=0)

        # observed context (within the zoom window): black dots + grey connector.
        # Connect only consecutive observed samples so the connector never bridges
        # the gap (no straight diagonal across the shaded region).
        obs_win = obs & win
        widx = np.where(obs_win)[0]
        if widx.size:
            breaks = np.where(np.diff(widx) > 1)[0] + 1
            for seg in np.split(widx, breaks):
                ax.plot(x_rel[seg], truth[seg], color="0.78", lw=0.7, zorder=1)
            ax.scatter(
                x_rel[widx], truth[widx], s=10.0, color="black", zorder=8, linewidths=0
            )

        # ground truth inside the gap: thick dark dashed line
        ax.plot(
            x_rel[span], truth[span], color="0.10", lw=2.2, ls=(0, (3.5, 2)), zorder=4
        )

        # per-rung imputed curve inside the gap: white halo then coloured line.
        # control gets a thicker neutral reference line; added-modality rungs are
        # thinner vivid lines on top.
        for p in prefixes:
            pred = z2mgdl(d[f"p{i}_pred_{p}"])
            ax.plot(
                x_rel[span],
                pred[span],
                color="white",
                lw=2.6,
                alpha=0.6,
                zorder=5,
                solid_capstyle="round",
            )
        for p in prefixes:
            pred = z2mgdl(d[f"p{i}_pred_{p}"])
            is_ctrl = p == CONTROL_PREFIX
            ax.plot(
                x_rel[span],
                pred[span],
                color=rung_color[p],
                lw=2.4 if is_ctrl else 1.5,
                alpha=0.95,
                zorder=6 if is_ctrl else 7,
                solid_capstyle="round",
            )

        # zoom x-limits
        ax.set_xlim(x_rel[lo], x_rel[hi])

        # y-limits auto-scaled to everything in the zoom window
        vals = [truth[span]]
        if widx.size:
            vals.append(truth[widx])
        for p in prefixes:
            vals.append(z2mgdl(d[f"p{i}_pred_{p}"])[span])
        allv = np.concatenate(vals)
        ymin, ymax = float(np.nanmin(allv)), float(np.nanmax(allv))
        rng_y = max(ymax - ymin, 6.0)
        ax.set_ylim(ymin - 0.12 * rng_y, ymax + 0.12 * rng_y)

        # ticks: a couple of clean x ticks (relative minutes), modest y ticks
        ax.tick_params(pad=2.0)
        ax.locator_params(axis="x", nbins=4)
        ax.locator_params(axis="y", nbins=4)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.8)

        # panel letter (bold, top-left, just outside)
        ax.text(
            -0.02,
            1.13,
            panel_letters[k],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14.0,
            fontweight="bold",
        )

        # strategy label on the leftmost panel of each row (row label)
        if c == 0:
            ax.set_ylabel(
                STRAT_TITLE[m["strategy"]] + "\nglucose (mg/dL)",
                fontsize=12.0,
                labelpad=4,
            )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="0.78",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=6.0,
            lw=0.8,
            label="observed CGM",
        ),
        Line2D(
            [0], [0], color="0.10", lw=2.2, ls=(0, (3.5, 2)), label="ground truth (gap)"
        ),
    ]
    for p in prefixes:
        is_ctrl = p == CONTROL_PREFIX
        handles.append(
            Line2D(
                [0],
                [0],
                color=rung_color[p],
                lw=3.4 if is_ctrl else 2.6,
                label=RUNG_LABEL.get(p, p),
            )
        )

    # single shared x-axis label centred beneath the grid (avoids 5 overlapping
    # per-panel labels)
    fig.supxlabel("time from gap start (min)", fontsize=12.5, y=0.150)

    # one shared legend beneath the grid, large font, named modalities.
    # 4 columns x 3 rows keeps every label fully inside the figure margins.
    leg = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.006),
        handlelength=2.2,
        columnspacing=2.2,
        labelspacing=0.7,
        fontsize=12.5,
    )
    leg.set_title(
        "conditioning modality (cumulative, CAIR ladder)",
        prop={"size": 13.0, "weight": "bold"},
    )

    fig.subplots_adjust(
        left=0.075, right=0.99, top=0.965, bottom=0.205, wspace=0.34, hspace=0.42
    )

    png = os.path.join(FIGDIR, "imputation_ladder_comparison.png")
    pdf = os.path.join(FIGDIR, "imputation_ladder_comparison.pdf")
    fig.savefig(png, dpi=330)
    fig.savefig(pdf)
    plt.close(fig)
    print("saved:", png)
    print("saved:", pdf)

    md_path = os.path.join(FIGDIR, "imputation_ladder_comparison_rmse.md")
    csv_path = os.path.join(FIGDIR, "imputation_ladder_comparison_rmse.csv")

    rows = []
    for k, m in enumerate(ordered):
        rmse = m["rmse"]
        best_p = min(prefixes, key=lambda p: rmse[p])
        rows.append(
            dict(
                panel=panel_letters[k],
                strategy=m["strategy"],
                participant=m["person_id"],
                rmse={p: rmse[p] for p in prefixes},
                best=best_p,
            )
        )

    # CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = (
            ["panel", "strategy", "participant_id"]
            + [f"{p}_rmse_mgdl" for p in prefixes]
            + ["best_rung"]
        )
        w.writerow(header)
        for row in rows:
            w.writerow(
                [row["panel"], row["strategy"], row["participant"]]
                + [f"{row['rmse'][p]:.2f}" for p in prefixes]
                + [row["best"]]
            )
    print("saved:", csv_path)

    # Markdown (best rung per row in bold)
    with open(md_path, "w") as f:
        f.write("# In-gap RMSE (mg/dL) per panel per CAIR ladder rung\n\n")
        f.write(
            "Each value is the RMSE between the imputed and held-out ground "
            "truth inside the zoomed (longest-contiguous) gap of the panel. "
            "Predictions are the mean of a 5-member seed ensemble. The "
            "lowest RMSE in each row is shown in **bold**. Column headers "
            "give the cumulative number of conditioning modalities in "
            "parentheses.\n\n"
        )
        head = (
            ["panel", "strategy", "participant"]
            + [f"{RUNG_LABEL.get(p, p)} ({cond_count[p]})" for p in prefixes]
            + ["best rung"]
        )
        f.write("| " + " | ".join(head) + " |\n")
        f.write("|" + "|".join(["---"] * len(head)) + "|\n")
        for row in rows:
            cells = [
                row["panel"],
                STRAT_TITLE[row["strategy"]],
                str(row["participant"]),
            ]
            for p in prefixes:
                v = f"{row['rmse'][p]:.1f}"
                if p == row["best"]:
                    v = f"**{v}**"
                cells.append(v)
            cells.append(RUNG_LABEL.get(row["best"], row["best"]))
            f.write("| " + " | ".join(cells) + " |\n")
        # per-strategy mean row and global mean
        f.write("\n## Mean in-gap RMSE by strategy\n\n")
        f.write(
            "| strategy | "
            + " | ".join(f"{RUNG_LABEL.get(p, p)} ({cond_count[p]})" for p in prefixes)
            + " |\n"
        )
        f.write("|" + "|".join(["---"] * (len(prefixes) + 1)) + "|\n")
        for s in STRAT_ROW:
            srows = [r for r in rows if r["strategy"] == s]
            means = {p: np.mean([r["rmse"][p] for r in srows]) for p in prefixes}
            bestp = min(prefixes, key=lambda p: means[p])
            cells = [STRAT_TITLE[s]]
            for p in prefixes:
                v = f"{means[p]:.1f}"
                if p == bestp:
                    v = f"**{v}**"
                cells.append(v)
            f.write("| " + " | ".join(cells) + " |\n")
        allmeans = {p: np.mean([r["rmse"][p] for r in rows]) for p in prefixes}
        bestp = min(prefixes, key=lambda p: allmeans[p])
        cells = ["**all (25 panels)**"]
        for p in prefixes:
            v = f"{allmeans[p]:.1f}"
            if p == bestp:
                v = f"**{v}**"
            cells.append(v)
        f.write("| " + " | ".join(cells) + " |\n")
    print("saved:", md_path)

    pid_by_row = {
        s: [r["participant"] for r in rows if r["strategy"] == s] for s in STRAT_ROW
    }
    cap_path = os.path.join(FIGDIR, "imputation_ladder_comparison_caption.txt")
    cmin, cmax = min(cond_count.values()), max(cond_count.values())
    rung_span = f"{cmin} to {cmax}"
    modality_list = "; ".join(
        f"{RUNG_LABEL.get(p, p)} ({cond_count[p]})" for p in prefixes
    )
    caption = (
        "Figure | Effect of progressive multimodal conditioning on CGM gap "
        "imputation across the CAIR modality ladder. A 5 x 5 grid of carved "
        "gaps is shown: each row is one physiological masking strategy "
        "(row 1 overnight/sleep; row 2 ascending/rise; row 3 post-meal; "
        "row 4 dipping/fall; row 5 mixed/combined) and each column is a distinct "
        "AI-READI participant, giving five independent examples per gap type "
        "(25 panels, a-y). Gaps are carved within the day-2 window with the "
        "physiological masker (target 20%); each panel is zoomed tightly to its "
        "longest contiguous gap (gap span plus a 3-sample margin) so the "
        "per-rung imputations fill the panel and are visually separable. In "
        "every panel the observed CGM context is drawn as black dots (thin grey "
        "connector), the gap is shaded grey with the held-out ground truth as a "
        "dark dashed line, and the imputed curve of each ladder rung is overlaid "
        "in a distinct colourblind-safe (Okabe-Ito) colour with a thin white "
        "halo for separation. The unconditioned control is a thick neutral "
        "dark-grey reference line; each added-modality rung is a distinct vivid "
        "colour. Rungs (cumulative conditioning modalities, "
        + rung_span
        + " channels): "
        + modality_list
        + ". No RMSE text is printed on the "
        "panels; per-panel, per-rung in-gap RMSE (mg/dL) is tabulated separately "
        "in figs/imputation_ladder_comparison_rmse.md (and .csv). Each rung's "
        "prediction is the mean of a 5-member seed ensemble. Participants by row "
        "-- sleep: {sleep}; ascending: {asc}; meal_post: {meal}; dipping: {dip}; "
        "combined: {comb}."
    ).format(
        sleep=", ".join(map(str, pid_by_row["sleep"])),
        asc=", ".join(map(str, pid_by_row["ascending"])),
        meal=", ".join(map(str, pid_by_row["meal_post"])),
        dip=", ".join(map(str, pid_by_row["dipping"])),
        comb=", ".join(map(str, pid_by_row["combined"])),
    )
    with open(cap_path, "w") as f:
        f.write(caption + "\n")
    print("saved:", cap_path)


if __name__ == "__main__":
    main()
