"""Canonical multimodal channel layout. Single source of truth for prep/train/eval."""

from __future__ import annotations

# - per-timestep cond modalities: name -> (record_value_key, n_value_channels) -
# The matching presence key is "<record_value_key>_p" (1 channel).
TS_MODALITIES = {
    "hr": ("mm_hr", 1),  # heart rate value (z-normed)
    "steps": ("mm_steps", 1),  # steps per 5-min (z-normed)
    "cal": ("mm_cal", 1),  # kcal per 5-min (z-normed)
    "sleep": ("mm_sleep", 4),  # one-hot {awake,light,deep,rem}
    "resp": ("mm_resp", 1),  # respiratory rate (z-normed)
    "stress": ("mm_stress", 1),  # stress level (z-normed)
    "env": ("mm_env", 2),  # ambient light intensity, temperature (z-normed)
    "spo2": ("mm_spo2", 1),  # SpO2 pulse oximetry (z-normed) [MIMIC vital]
    "abp": ("mm_abp", 1),  # mean arterial blood pressure (z-normed) [MIMIC vital]
}

# ladder rung -> active per-timestep modalities (cumulative)
TS_RUNGS = {
    0: [],
    1: ["hr"],
    2: ["hr", "steps", "cal"],
    3: ["hr", "steps", "cal", "sleep"],
    4: ["hr", "steps", "cal", "sleep", "resp", "stress"],
    5: ["hr", "steps", "cal", "sleep", "resp", "stress", "env"],
}

# - per-subject static blocks: name -> (record_key, n_embed_dims) -
# The matching presence key is "<record_key>_p" (1 scalar).
STATIC_BLOCKS = {
    "clinical": (
        "mm_clinical",
        8,
    ),  # age (z) + diabetes-group one-hot + site, padded to 8
    "ecg": (
        "mm_ecg_emb",
        8,
    ),  # 12-lead manifest scalars: Rate,PR,QRSD,QT,QTc,P,QRS,T (z)
    "retinal": (
        "mm_retinal_emb",
        128,
    ),  # frozen DINOv2 [cfp|oct|octa|flio] embedding, PCA-reduced
}

# ladder rung -> active static blocks (cumulative; layered on top of full TS set)
STATIC_RUNGS = {
    6: ["clinical"],
    7: ["clinical", "ecg"],
    8: ["clinical", "ecg", "retinal"],
}


def ts_width(active) -> int:
    """K: total per-timestep cond columns (value channels + 1 presence each)."""
    return sum(TS_MODALITIES[m][1] + 1 for m in active)


def ctx_width(active) -> int:
    """S: total static-context dims (embedding dims + 1 presence each)."""
    return sum(STATIC_BLOCKS[b][1] + 1 for b in active)


def rung_active(rung: int):
    """Return (active_ts_modalities, active_static_blocks) for a ladder rung.

    Static rungs (>=6) include the full time-series set (rung 5) underneath.
    """
    if rung <= 5:
        return list(TS_RUNGS[rung]), []
    return list(TS_RUNGS[5]), list(STATIC_RUNGS[rung])


def parse_csv(s):
    """Parse a comma-separated modality/block list ('' -> [])."""
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def build_mod_and_ctx(s, ts_mods, static_blocks):
    """Build the per-timestep modality block and the static-context vector for a
    record dict `s`. Shared by the training dataset and the eval harness so column
    order and presence handling never drift.

    Returns (mod (T,K) float32 or None, ctx (S,) float32 or None).
    """
    import numpy as np

    ts_mods = [m for m in TS_MODALITIES if m in ts_mods]
    static_blocks = [b for b in STATIC_BLOCKS if b in static_blocks]
    T = int(np.asarray(s["irg_ts"]).shape[0])
    mod = None
    if ts_mods:
        cols = []
        for m in ts_mods:
            key, nval = TS_MODALITIES[m]
            cols.append(np.asarray(s[key], np.float32).reshape(T, nval))
            cols.append(np.asarray(s[key + "_p"], np.float32).reshape(T, 1))
        mod = np.concatenate(cols, axis=1).astype(np.float32)
    ctx = None
    if static_blocks:
        vec = []
        for b in static_blocks:
            key, ndim = STATIC_BLOCKS[b]
            if key in s:
                v = np.asarray(s[key], np.float32).reshape(-1)[:ndim]
                if len(v) < ndim:
                    v = np.concatenate([v, np.zeros(ndim - len(v), np.float32)])
                p = float(np.asarray(s.get(key + "_p", 1.0)).reshape(-1)[0])
            else:
                v = np.zeros(ndim, np.float32)
                p = 0.0
            vec.append(v)
            vec.append(np.array([p], np.float32))
        ctx = np.concatenate(vec).astype(np.float32)
    return mod, ctx
