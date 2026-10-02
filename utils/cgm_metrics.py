"""Extended, clinically-correct CGM metric battery for downstream metric-recovery."""

from __future__ import annotations

import numpy as np

CGM_MEAN = 132.05
CGM_STD = 42.33


def to_mgdl(z: np.ndarray) -> np.ndarray:
    return np.asarray(z, dtype=np.float64) * CGM_STD + CGM_MEAN


def mean_glucose(g):
    return float(np.mean(g))


def sd_glucose(g):
    return float(np.std(g, ddof=1)) if len(g) > 1 else 0.0


def cv_glucose(g):
    m = np.mean(g)
    return (
        float(100.0 * (np.std(g, ddof=1) if len(g) > 1 else 0.0) / m)
        if m > 1
        else float("nan")
    )


def tir(g):
    return float(np.mean((g >= 70) & (g <= 180)) * 100)


def tbr70(g):
    return float(np.mean(g < 70) * 100)


def tbr54(g):
    return float(np.mean(g < 54) * 100)


def tar180(g):
    return float(np.mean(g > 180) * 100)


def tar250(g):
    return float(np.mean(g > 250) * 100)


def gmi(g):
    return float(3.31 + 0.02392 * np.mean(g))


def _turning_points(g):
    """Indices of interior local extrema (peaks/nadirs), robust to plateaus.

    Compress runs of equal values, then keep interior points where the discrete slope
    changes sign. Global endpoints are NOT counted as extrema, so only complete
    nadir<->peak excursions are measured (standard Service-Nelson; avoids boundary
    half-excursions inflating/deflating MAGE).
    """
    n = len(g)
    keep = [0]
    for i in range(1, n):
        if g[i] != g[keep[-1]]:
            keep.append(i)
    if len(keep) < 3:
        return []
    gk = g[keep]
    turns = []
    for j in range(1, len(keep) - 1):
        if (gk[j] - gk[j - 1]) * (gk[j + 1] - gk[j]) < 0:
            turns.append(keep[j])
    return turns


def mage(g):
    """Faithful Service & Nelson (1970) MAGE.

    Excursions are nadir->peak (up) and peak->nadir (down) amplitudes between alternating
    turning points that exceed 1 SD of the day's glucose. MAGE is the mean of excursions
    counted in the direction of the FIRST qualifying excursion (MAGE+ or MAGE-), as in the
    original definition. Returns the same-direction mean in mg/dL.
    """
    g = np.asarray(g, dtype=np.float64)
    n = len(g)
    if n < 3:
        return float("nan")
    sd = np.std(g, ddof=1)
    if sd == 0:
        return 0.0
    ti = _turning_points(g)
    if len(ti) < 2:
        return 0.0
    ext = g[np.asarray(ti, dtype=np.int64)]
    deltas = np.diff(ext)  # signed excursion amplitudes
    qual = deltas[np.abs(deltas) > sd]  # exceed 1 SD
    if qual.size == 0:
        return 0.0
    first_dir = np.sign(qual[0])  # +1 = ascending excursions first
    same = qual[np.sign(qual) == first_dir]
    return float(np.mean(np.abs(same))) if same.size else float(np.mean(np.abs(qual)))


def _risk_transform(g):
    g = np.clip(np.asarray(g, dtype=np.float64), 20.0, 600.0)
    f = 1.509 * (np.power(np.log(g), 1.084) - 5.381)
    return f


def lbgi(g):
    f = _risk_transform(g)
    rl = np.where(f < 0, 10.0 * f * f, 0.0)
    return float(np.mean(rl))


def hbgi(g):
    f = _risk_transform(g)
    rh = np.where(f > 0, 10.0 * f * f, 0.0)
    return float(np.mean(rh))


def j_index(g):
    m = np.mean(g)
    s = np.std(g, ddof=1) if len(g) > 1 else 0.0
    return float(0.001 * (m + s) ** 2)


def _gri_components(g):
    vlow = np.mean(g < 54) * 100
    low = np.mean((g >= 54) & (g < 70)) * 100
    vhigh = np.mean(g > 250) * 100
    high = np.mean((g > 180) & (g <= 250)) * 100
    chypo = 3.0 * vlow + 2.4 * low  # hypoglycemia component
    chyper = 1.6 * vhigh + 0.8 * high  # hyperglycemia component
    return chypo, chyper


def gri(g):
    chypo, chyper = _gri_components(g)
    return float(min(max(chypo + chyper, 0.0), 100.0))


def gri_hypo(g):
    return float(_gri_components(g)[0])


def gri_hyper(g):
    return float(_gri_components(g)[1])


def auc_over_180(g):
    return float(np.mean(np.clip(np.asarray(g, dtype=np.float64) - 180.0, 0.0, None)))


METRIC_SPECS = [
    ("mean", mean_glucose, "mg/dL", "level"),
    ("gmi", gmi, "%", "level"),
    ("sd", sd_glucose, "mg/dL", "variability"),
    ("cv", cv_glucose, "%", "variability"),
    ("mage", mage, "mg/dL", "variability"),
    ("jindex", j_index, "a.u.", "variability"),
    ("tir", tir, "%", "level"),
    ("tbr70", tbr70, "%", "hypo"),
    ("tbr54", tbr54, "%", "hypo"),
    ("lbgi", lbgi, "a.u.", "hypo"),
    ("tar180", tar180, "%", "hyper"),
    ("tar250", tar250, "%", "hyper"),
    ("hbgi", hbgi, "a.u.", "hyper"),
    ("auc180", auc_over_180, "mg/dL", "hyper"),
    ("gri", gri, "0-100", "risk"),
    ("gri_hypo", gri_hypo, "0-100", "hypo"),
    ("gri_hyper", gri_hyper, "0-100", "hyper"),
]

# Clinically-meaningful-difference (MCID) per metric for agreement-rate reporting.
# Well-anchored: TIR 5% (Battelino 2019); GMI/eA1c 0.3% (ADA/Bergenstal). Others are
# defensible round values used only for the within-MCID agreement-rate figure.
MCID = {
    "mean": 10.0,
    "gmi": 0.3,
    "sd": 5.0,
    "cv": 2.0,
    "mage": 10.0,
    "jindex": 2.0,
    "tir": 5.0,
    "tbr70": 1.0,
    "tbr54": 0.5,
    "lbgi": 1.0,
    "tar180": 5.0,
    "tar250": 5.0,
    "hbgi": 2.0,
    "auc180": 5.0,
    "gri": 5.0,
    "gri_hypo": 5.0,
    "gri_hyper": 5.0,
}
# Anchored (literature-grade) MCIDs to highlight in the paper
MCID_ANCHORED = {"tir", "gmi"}
METRIC_KEYS = [k for (k, *_rest) in METRIC_SPECS]
METRIC_UNITS = {k: u for (k, _f, u, _g) in METRIC_SPECS}
METRIC_GROUP = {k: g for (k, _f, _u, g) in METRIC_SPECS}
# Pretty labels for figures
METRIC_LABELS = {
    "mean": "Mean",
    "gmi": "GMI",
    "sd": "SD",
    "cv": "CV",
    "mage": "MAGE",
    "jindex": "J-index",
    "tir": "TIR",
    "tbr70": "TBR<70",
    "tbr54": "TBR<54",
    "lbgi": "LBGI",
    "tar180": "TAR>180",
    "tar250": "TAR>250",
    "hbgi": "HBGI",
    "auc180": "AUC>180",
    "gri": "GRI",
    "gri_hypo": "GRI-Hypo",
    "gri_hyper": "GRI-Hyper",
}


def metric_vector_mgdl(g_mgdl) -> dict:
    g = np.asarray(g_mgdl, dtype=np.float64)
    return {k: fn(g) for (k, fn, _u, _grp) in METRIC_SPECS}


def metric_vector_z(g_z) -> dict:
    return metric_vector_mgdl(to_mgdl(g_z))
