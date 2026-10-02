"""Publication figures for the MIMIC clinical-ICU section."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "figs", "mimic")
os.makedirs(OUT, exist_ok=True)

PCTS = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3"]
PCT_F = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
MECHS = ["mcar", "mar", "nmar"]
MECH_LABEL = {"mcar": "MCAR", "mar": "MAR", "nmar": "NMAR"}
MRR_KEYS = ["mrr_tir", "mrr_tar180", "mrr_tbr70", "mrr_mage", "mrr_cv"]
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "figure.dpi": 150,
    }
)


def _agg(path):
    return json.load(open(path))["aggregate"]


def _rmse(agg, m, mech):
    v = [agg[f"{m}|{mech}|{p}"]["rmse"] for p in PCTS if f"{m}|{mech}|{p}" in agg]
    return float(np.mean(v)) if v else float("nan")


def _rmse_by_rate(agg, m, mech):
    return [agg.get(f"{m}|{mech}|{p}", {}).get("rmse", np.nan) for p in PCTS]


def _mrr(agg, m, mech):
    v = [
        agg[f"{m}|{mech}|{p}"][k]
        for p in PCTS
        for k in MRR_KEYS
        if f"{m}|{mech}|{p}" in agg
        and agg[f"{m}|{mech}|{p}"].get(k) == agg[f"{m}|{mech}|{p}"].get(k)
    ]
    return float(np.mean(v)) if v else float("nan")


THRESHOLD_KEYS = [
    "mrr_tir",
    "mrr_tar180",
    "mrr_tbr70",
]  # clinical threshold-crossing metrics


def _tmrr(agg, m, mech):
    v = [
        agg[f"{m}|{mech}|{p}"][k]
        for p in PCTS
        for k in THRESHOLD_KEYS
        if f"{m}|{mech}|{p}" in agg
        and agg[f"{m}|{mech}|{p}"].get(k) == agg[f"{m}|{mech}|{p}"].get(k)
    ]
    return float(np.mean(v)) if v else float("nan")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.png/.pdf")


# Figure 1: 12-method reconstruction RMSE by mechanism (ABP + HR)


def fig1_benchmark():
    abp = _agg(os.path.join(RES, "toye_mimic_abp_realistic_full_n200.json"))
    hr = _agg(os.path.join(RES, "toye_mimic_hr_realistic_full_n200.json"))
    methods = [
        ("linear", "Linear", "#4c72b0"),
        ("knn", "k-NN", "#8fa8d0"),
        ("mice", "MICE", "#b0c4de"),
        ("gpvae", "GP-VAE", "#c44e52"),
        ("mrnn", "MRNN", "#e07b7b"),
        ("cair_mm", "CAIR (ours)", "#2ca02c"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, agg, title, unit in (
        (axes[0], abp, "MIMIC-III ABP", "mmHg"),
        (axes[1], hr, "MIMIC-III HR", "bpm"),
    ):
        x = np.arange(len(MECHS))
        w = 0.13
        for i, (m, lab, c) in enumerate(methods):
            vals = [_rmse(agg, m, mech) for mech in MECHS]
            ax.bar(
                x + (i - 2.5) * w,
                vals,
                w,
                label=lab,
                color=c,
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([MECH_LABEL[m] for m in MECHS])
        ax.set_ylabel(f"Reconstruction RMSE ({unit})")
        ax.set_title(title)
    axes[0].legend(ncol=3, fontsize=8.5, loc="upper left", framealpha=0.9)
    _save(fig, "fig1_benchmark_rmse")


# Figure 2: training-augmentation ablation (physio vs realistic) on ABP


def fig2_ablation():
    old = _agg(os.path.join(RES, "toye_mimic_abp_n200.json"))  # physio CAIR + baselines
    new = _agg(os.path.join(RES, "toye_mimic_abp_realistic_full_n200.json"))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(MECHS))
    w = 0.26
    lin = [_rmse(new, "linear", mech) for mech in MECHS]
    phys = [
        min(_rmse(old, "cair", mech), _rmse(old, "cair_mm", mech)) for mech in MECHS
    ]
    real = [
        min(_rmse(new, "cair", mech), _rmse(new, "cair_mm", mech)) for mech in MECHS
    ]
    ax.bar(
        x - w,
        lin,
        w,
        label="Linear interp",
        color="#4c72b0",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x,
        phys,
        w,
        label="CAIR (CGM-mask training)",
        color="#dd8452",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x + w,
        real,
        w,
        label="CAIR (realistic-gap training)",
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.4,
    )
    for xi, (p, r) in enumerate(zip(phys, real)):
        ax.annotate(
            "",
            xy=(xi + w, r),
            xytext=(xi, p),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([MECH_LABEL[m] for m in MECHS])
    ax.set_ylabel("ABP reconstruction RMSE (mmHg)")
    ax.set_title("Training-mask alignment makes CAIR win (MIMIC-III ABP)")
    ax.legend(fontsize=9, loc="upper left")
    _save(fig, "fig2_mask_ablation")


# Figure 3: RMSE vs missingness rate, ABP, three mechanisms


def fig3_rate_curves():
    agg = _agg(os.path.join(RES, "toye_mimic_abp_realistic_full_n200.json"))
    methods = [
        ("cair_mm", "CAIR (ours)", "#2ca02c", "o", "-"),
        ("linear", "Linear", "#4c72b0", "s", "-"),
        ("knn", "k-NN", "#937860", "^", "--"),
        ("gpvae", "GP-VAE", "#c44e52", "v", ":"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, mech in zip(axes, MECHS):
        for m, lab, c, mk, ls in methods:
            ax.plot(
                [p * 100 for p in PCT_F],
                _rmse_by_rate(agg, m, mech),
                marker=mk,
                color=c,
                ls=ls,
                label=lab,
                ms=5,
            )
        ax.set_title(MECH_LABEL[mech])
        ax.set_xlabel("Missingness rate (%)")
    axes[0].set_ylabel("ABP reconstruction RMSE (mmHg)")
    axes[0].legend(fontsize=8.5, loc="upper left")
    _save(fig, "fig3_rate_curves_abp")


# Figure 4: downstream dissociation - burden MRR (imputation matters) vs
#           clinical-outcome AUROC (imputation-agnostic), ABP


def fig4_downstream():
    agg = _agg(os.path.join(RES, "toye_mimic_abp_realistic_full_n200.json"))
    clin = json.load(open(os.path.join(RES, "mimic_clinical_downstream_abp.json")))[
        "results"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) clinical-burden MRR by mechanism
    ax = axes[0]
    bmethods = [
        ("linear", "Linear", "#4c72b0"),
        ("knn", "k-NN", "#937860"),
        ("cair_mm", "CAIR (ours)", "#2ca02c"),
    ]
    x = np.arange(len(MECHS))
    w = 0.26
    for i, (m, lab, c) in enumerate(bmethods):
        ax.bar(
            x + (i - 1) * w,
            [_tmrr(agg, m, mech) for mech in MECHS],
            w,
            label=lab,
            color=c,
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([MECH_LABEL[m] for m in MECHS])
    ax.set_ylabel("Clinical-burden MRR (time-in/above/below-range)")
    ax.set_title("(a) Burden recovery: CAIR and k-NN vs linear")
    ax.legend(fontsize=9, loc="upper right")

    # (b) single-vital mortality AUROC vs NMAR rate: imputation matters, grows with missingness
    ax = axes[1]
    sweep = json.load(open(os.path.join(RES, "mimic_singlevital_abp_sweep.json")))[
        "results"
    ]
    srates = sorted({r["rate"] for r in sweep})
    smeth = [
        ("real", "Oracle fill", "#000000", "o", "-"),
        ("cair_mm", "CAIR (ours)", "#2ca02c", "s", "-"),
        ("linear", "Linear", "#4c72b0", "^", "--"),
        ("mean", "Mean-fill", "#bbbbbb", "v", ":"),
    ]

    def sv(method, rate):
        hit = [
            r
            for r in sweep
            if r["label"] == "mortality"
            and r["method"] == method
            and abs(r["rate"] - rate) < 1e-6
        ]
        return hit[0]["auroc"] if hit and hit[0]["auroc"] is not None else np.nan

    for m, lab, c, mk, ls in smeth:
        ax.plot(
            [r * 100 for r in srates],
            [sv(m, rt) for rt in srates],
            marker=mk,
            color=c,
            ls=ls,
            label=lab,
            ms=7,
            lw=2,
        )
    ax.set_xlabel("NMAR missingness rate (%)")
    ax.set_ylabel("Mortality AUROC (single-vital ABP)")
    ax.set_title("(b) Hard outcome: imputation matters, grows with missingness")
    ax.legend(fontsize=8.5, loc="lower left")
    _save(fig, "fig4_downstream_dissociation")


if __name__ == "__main__":
    fig1_benchmark()
    fig2_ablation()
    fig3_rate_curves()
    fig4_downstream()
    print("all figures written to", OUT)
