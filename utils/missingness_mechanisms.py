#!/usr/bin/env python3
"""Missingness-mechanism simulators (Toye et al., CHIL 2025)."""

from __future__ import annotations

import numpy as np


def _finalize(original_mask: np.ndarray, deleted: np.ndarray):
    """Given a boolean (L,) 'deleted' array, build (eval_mask, target_mask)."""
    L = original_mask.shape[0]
    obs = original_mask[:, 0].astype(bool)
    deleted = deleted & obs  # never delete an already-missing point
    eval_mask = original_mask.copy()
    eval_mask[deleted, 0] = 0.0
    target_mask = np.zeros((L, 1), dtype=np.float32)
    target_mask[deleted, 0] = 1.0
    return eval_mask, target_mask


def create_mcar_mask(
    series: np.ndarray, original_mask: np.ndarray, percent: float = 0.20, seed: int = 42
):
    """Independent Bernoulli deletion of observed points at rate `percent`."""
    rng = np.random.RandomState(seed)
    obs = original_mask[:, 0].astype(bool)
    idx = np.flatnonzero(obs)
    n_del = int(round(len(idx) * percent))
    deleted = np.zeros(series.shape[0], dtype=bool)
    if n_del > 0:
        chosen = rng.choice(idx, size=n_del, replace=False)
        deleted[chosen] = True
    return _finalize(original_mask, deleted)


def native_gap_durations(mask: np.ndarray) -> list[int]:
    """Lengths of contiguous missing (mask==0) runs in a (L,1) observed mask.

    Used to sample realistic gap durations for MAR/NMAR, per Toye et al."""
    missing = (mask[:, 0] == 0).astype(np.int8)
    durs = []
    run = 0
    for v in missing:
        if v == 1:
            run += 1
        elif run > 0:
            durs.append(run)
            run = 0
    if run > 0:
        durs.append(run)
    return durs


_DEFAULT_GAP_DURATIONS = [3, 6, 9, 12]  # 15/30/45/60 min at a 5-min grid


def _windowed_delete(
    series, original_mask, trigger, percent, gap_durations, seed, high_is_missing=True
):
    """Delete contiguous windows anchored at extreme trigger values until the
    `percent` budget of observed points is spent.

    Anchors are observed positions ranked by trigger (descending if
    high_is_missing, else by a caller-provided pre-sorted trigger). Each anchor
    grows a window of a sampled duration centered on it; windows are clipped to
    observed positions."""
    rng = np.random.RandomState(seed)
    L = series.shape[0]
    obs = original_mask[:, 0].astype(bool)
    n_obs = int(obs.sum())
    budget = int(round(n_obs * percent))
    durations = list(gap_durations) if gap_durations else list(_DEFAULT_GAP_DURATIONS)

    order = np.argsort(-trigger)  # highest trigger first
    order = [i for i in order if obs[i]]

    deleted = np.zeros(L, dtype=bool)
    for anchor in order:
        if deleted.sum() >= budget:
            break
        dur = int(durations[rng.randint(len(durations))])
        half = dur // 2
        lo = max(0, anchor - half)
        hi = min(L, lo + dur)
        deleted[lo:hi] = True
    # trim overshoot back toward budget by unsetting from the last-added region
    if deleted.sum() > budget:
        on = np.flatnonzero(deleted)
        for i in on[::-1]:
            if deleted.sum() <= budget:
                break
            deleted[i] = False
    return _finalize(original_mask, deleted)


def create_mar_mask(
    series, original_mask, trigger, percent=0.20, gap_durations=None, seed=42
):
    """MAR: deletion triggered by a high external covariate `trigger` (L,)."""
    trigger = np.asarray(trigger, dtype=np.float32).reshape(-1)
    return _windowed_delete(
        series, original_mask, trigger, percent, gap_durations, seed
    )


def create_nmar_mask(
    series,
    original_mask,
    percent=0.20,
    gap_durations=None,
    seed=42,
    mean_mgdl=132.05,
    std_mgdl=42.33,
    lo_mgdl=70.0,
    hi_mgdl=150.0,
):
    """NMAR: deletion triggered by the glucose value itself (< lo or > hi mg/dL).

    Builds a trigger that is large where |value| is clinically extreme, then
    reuses the same windowing as MAR."""
    g = series[:, 0].astype(np.float32) * std_mgdl + mean_mgdl
    # distance outside the [lo, hi] band; 0 inside, positive outside
    below = np.clip(lo_mgdl - g, 0, None)
    above = np.clip(g - hi_mgdl, 0, None)
    trigger = (below + above).astype(np.float32)
    # add a tiny deterministic jitter so all-inside-band still ranks stably
    rng = np.random.RandomState(seed)
    trigger = trigger + 1e-6 * rng.rand(len(trigger)).astype(np.float32)
    return _windowed_delete(
        series, original_mask, trigger, percent, gap_durations, seed
    )


def create_realistic_mask(
    series: np.ndarray,
    original_mask: np.ndarray,
    target_ratio: float = 0.20,
    seed: int = 42,
    min_gap: int = 2,
    max_gap: int = 24,
):
    """Generic realistic-missingness augmentation for TRAINING (not the eval protocol).

    Mixes MCAR scatter with random contiguous gaps of varied length at random observed
    positions, deleting up to `target_ratio` of the observed points. Deliberately
    domain-agnostic: it uses NO covariate or value triggers, so it does not encode the
    MAR/NMAR benchmark mechanisms. It only makes an imputer robust to gaps of arbitrary
    shape and position, which is a fair training-time choice (train split only; the eval
    masks and their seeds are never seen). Returns (eval_mask, target_mask) shaped (L,1),
    same contract as the eval generators.
    """
    rng = np.random.RandomState(seed)
    L = series.shape[0]
    obs = original_mask[:, 0].astype(bool)
    obs_idx = np.flatnonzero(obs)
    deleted = np.zeros(L, dtype=bool)
    budget = int(round(len(obs_idx) * target_ratio))
    if budget <= 0 or len(obs_idx) == 0:
        return _finalize(original_mask, deleted)
    if rng.rand() < 0.4:
        # MCAR scatter
        chosen = rng.choice(obs_idx, size=min(budget, len(obs_idx)), replace=False)
        deleted[chosen] = True
    else:
        # random contiguous gaps of varied length anchored at random observed points
        guard = 0
        while deleted.sum() < budget and guard < 10 * L:
            guard += 1
            g = int(rng.randint(min_gap, max_gap + 1))
            anchor = obs_idx[rng.randint(len(obs_idx))]
            lo = max(0, anchor - g // 2)
            hi = min(L, lo + g)
            deleted[lo:hi] = True
        # trim overshoot back toward budget (only among observed positions)
        over = int((deleted & obs).sum()) - budget
        if over > 0:
            on = np.flatnonzero(deleted & obs)
            rng.shuffle(on)
            for i in on[:over]:
                deleted[i] = False
    return _finalize(original_mask, deleted)
