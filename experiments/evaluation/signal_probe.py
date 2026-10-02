#!/usr/bin/env python3
"""Signal probe: is contemporaneous HR / steps related to CGM dynamics?"""

from __future__ import annotations

from utils.paths import DATA_ROOT, REPO_ROOT, CACHE_ROOT

import json
import os
import pickle

import numpy as np

D = str(DATA_ROOT)
MM = D + "/aireadi_cgm_mm"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CGM_STD = 42.33


def _pearson(x, y):
    if len(x) < 30 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    recs = pickle.load(open(f"{MM}/aireadi_cgm_test.pkl", "rb"))
    cols = {
        "dcgm": [],
        "absd": [],
        "hr": [],
        "hr_p": [],
        "steps": [],
        "steps_p": [],
        "sleep_p": [],
        "resp": [],
        "resp_p": [],
    }
    for r in recs:
        cgm = r["irg_ts"][:, 0].astype(np.float32) * CGM_STD  # mg/dL
        obs = r["irg_ts_mask"][:, 0] > 0
        d = np.zeros_like(cgm)
        d[1:] = cgm[1:] - cgm[:-1]
        valid = obs.copy()
        valid[1:] &= obs[:-1]  # both endpoints observed
        for name, key in (("hr", "mm_hr"), ("steps", "mm_steps"), ("resp", "mm_resp")):
            cols[name].append(r[key][:, 0][valid])
            cols[name + "_p"].append(r[key + "_p"][:, 0][valid])
        cols["sleep_p"].append(r["mm_sleep_p"][:, 0][valid])
        cols["dcgm"].append(d[valid])
        cols["absd"].append(np.abs(d[valid]))
    for k in cols:
        cols[k] = np.concatenate(cols[k])

    out = {}
    # overall: |dCGM| vs HR/steps/resp (where the signal is present)
    for name in ("hr", "steps", "resp"):
        m = cols[name + "_p"] > 0
        out[f"absdCGM~{name}"] = _pearson(cols["absd"][m], cols[name][m])
        out[f"dCGM~{name}"] = _pearson(cols["dcgm"][m], cols[name][m])
    # sleep vs awake (sleep-stage present as proxy for night): |dCGM| level
    sl = cols["sleep_p"] > 0
    out["absdCGM_mean_sleep"] = (
        float(np.nanmean(cols["absd"][sl])) if sl.any() else float("nan")
    )
    out["absdCGM_mean_awake"] = (
        float(np.nanmean(cols["absd"][~sl])) if (~sl).any() else float("nan")
    )
    # HR vs |dCGM| within sleep and within awake
    for seg, mask in (("sleep", sl), ("awake", ~sl)):
        m = mask & (cols["hr_p"] > 0)
        out[f"absdCGM~hr[{seg}]"] = _pearson(cols["absd"][m], cols["hr"][m])
    # steps vs signed dCGM in the next-step (exercise tends to lower glucose)
    m = cols["steps_p"] > 0
    out["dCGM~steps[active]"] = _pearson(cols["dcgm"][m], cols["steps"][m])

    print("=== signal probe (mm test) ===")
    for k, v in out.items():
        print(f"  {k:24s} {v:+.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # figure
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        names = [k for k in out if k.startswith(("absdCGM~", "dCGM~"))]
        vals = [out[k] for k in names]
        ax[0].barh(names, vals, color="#3b7dd8")
        ax[0].axvline(0, color="k", lw=0.7)
        ax[0].set_title("Pearson r: wearable vs CGM dynamics")
        ax[0].set_xlabel("correlation")
        ax[1].bar(
            ["sleep", "awake"],
            [out["absdCGM_mean_sleep"], out["absdCGM_mean_awake"]],
            color=["#2a4d8f", "#d8943b"],
        )
        ax[1].set_title("|dCGM| (mg/dL per 5 min) by segment")
        ax[1].set_ylabel("mean |dCGM|")
        plt.tight_layout()
        os.makedirs(os.path.join(ROOT, "figs"), exist_ok=True)
        dst = os.path.join(ROOT, "figs", "signal_probe.png")
        plt.savefig(dst, dpi=130)
        print(f"saved -> {dst}")
    except Exception as e:
        print(f"[fig skipped: {e}]")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(
        out, open(os.path.join(ROOT, "results", "signal_probe.json"), "w"), indent=2
    )


if __name__ == "__main__":
    main()
