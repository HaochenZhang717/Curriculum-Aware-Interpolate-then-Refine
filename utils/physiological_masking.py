#!/usr/bin/env python3
"""Physiologically-motivated masking strategies for CGM imputation evaluation."""

from __future__ import annotations

import numpy as np

# Time-of-day window definitions

# Each tuple is (start_index_inclusive, end_index_exclusive) on L=288 grid.
MEAL_WINDOWS = {
    "breakfast": (84, 114),  # 7:00 - 9:30 AM  (post-prandial rise window)
    "lunch": (144, 174),  # 12:00 - 2:30 PM
    "dinner": (216, 252),  # 6:00 - 9:00 PM
}

# Sleep / nocturnal: midnight → 6 AM  and  11 PM → midnight
SLEEP_WINDOWS = [(0, 72), (276, 288)]


# Time-of-day flag arrays


def flag_meal_timesteps(L: int = 288) -> np.ndarray:
    """
    Boolean array of length L marking post-meal excursion windows.

    For multi-day sequences (L > 288), the daily pattern repeats every 288
    timesteps, since recordings always start at midnight.
    """
    flag = np.zeros(L, dtype=bool)
    idx = np.arange(L)
    tod = idx % 288  # time-of-day index within each 24-hour cycle
    for lo, hi in MEAL_WINDOWS.values():
        flag[(tod >= lo) & (tod < hi)] = True
    return flag


def flag_sleep_timesteps(L: int = 288) -> np.ndarray:
    """
    Boolean array of length L marking the nocturnal sleep period.

    Pattern repeats every 288 timesteps for multi-day sequences.
    """
    flag = np.zeros(L, dtype=bool)
    idx = np.arange(L)
    tod = idx % 288
    for lo, hi in SLEEP_WINDOWS:
        flag[(tod >= lo) & (tod < hi)] = True
    return flag


# Slope-based detectors (ascending / dipping)


def _smooth_series(series: np.ndarray, obs_mask: np.ndarray, hw: int = 2) -> np.ndarray:
    """
    Simple window-average of the observed signal, ignoring missing points.
    Returns a smoothed array of the same length.
    """
    L = len(series)
    out = series.copy()
    for i in range(L):
        lo = max(0, i - hw)
        hi = min(L, i + hw + 1)
        vals = series[lo:hi][obs_mask[lo:hi]]
        if len(vals) > 0:
            out[i] = vals.mean()
    return out


def _select_runs(active: np.ndarray, min_run: int) -> np.ndarray:
    """
    Given a boolean array, keep only contiguous True-runs of length >= min_run.
    Returns a new boolean array.
    """
    L = len(active)
    result = np.zeros(L, dtype=bool)
    i = 0
    while i < L:
        if active[i]:
            j = i + 1
            while j < L and active[j]:
                j += 1
            if (j - i) >= min_run:
                result[i:j] = True
            i = j
        else:
            i += 1
    return result


def flag_ascending_segments(
    series: np.ndarray,
    obs_mask: np.ndarray,
    slope_thresh: float = 0.05,
    min_run: int = 4,
    smooth_hw: int = 2,
) -> np.ndarray:
    """
    Flag timesteps belonging to a sustained glucose rise.

    Args:
        series      : 1-D normalized CGM values (length L)
        obs_mask    : 1-D boolean observed mask (length L)
        slope_thresh: per-timestep change (normalized) to count as 'rising'.
                      Default 0.05 ≈ 2-3 mg/dL per 5 min.
        min_run     : minimum consecutive timesteps to constitute a run.
        smooth_hw   : half-window for smoothing before slope computation.

    Returns:
        1-D boolean array, True where a sustained rise is detected.
    """
    if len(series) == 0:
        return np.zeros(0, dtype=bool)
    smoothed = _smooth_series(series, obs_mask, hw=smooth_hw)
    slope = np.diff(smoothed, prepend=smoothed[0])
    rising = (slope > slope_thresh) & obs_mask
    return _select_runs(rising, min_run=min_run)


def flag_dipping_segments(
    series: np.ndarray,
    obs_mask: np.ndarray,
    slope_thresh: float = 0.05,
    min_run: int = 4,
    smooth_hw: int = 2,
) -> np.ndarray:
    """
    Flag timesteps belonging to a sustained glucose drop.

    slope_thresh: magnitude threshold (drop if slope < -slope_thresh).
    """
    if len(series) == 0:
        return np.zeros(0, dtype=bool)
    smoothed = _smooth_series(series, obs_mask, hw=smooth_hw)
    slope = np.diff(smoothed, prepend=smoothed[0])
    dropping = (slope < -slope_thresh) & obs_mask
    return _select_runs(dropping, min_run=min_run)


# Block extraction helpers


def _extract_contiguous_blocks(
    flag: np.ndarray, obs: np.ndarray
) -> list[tuple[int, int]]:
    """
    Find contiguous True-runs in (flag & obs) and return as list of
    (start_inclusive, end_exclusive) pairs, sorted by length descending.
    """
    active = flag & obs
    blocks = []
    i = 0
    L = len(active)
    while i < L:
        if active[i]:
            j = i + 1
            while j < L and active[j]:
                j += 1
            blocks.append((i, j))
            i = j
        else:
            i += 1
    # Longest blocks first so we fill the budget efficiently.
    blocks.sort(key=lambda b: b[1] - b[0], reverse=True)
    return blocks


def _blocks_to_mask(
    blocks: list[tuple[int, int]],
    L: int,
    obs: np.ndarray,
    n_target: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Greedily select whole blocks (shuffled) until we exceed n_target masked
    points, then return the set of masked indices as a boolean array.
    """
    rng.shuffle(blocks)  # random ordering to avoid always picking first meal
    masked = np.zeros(L, dtype=bool)
    count = 0
    for lo, hi in blocks:
        if count >= n_target:
            break
        block_obs = np.sum(obs[lo:hi])
        masked[lo:hi] = True
        count += block_obs
    return masked


# Master mask builder


def create_physiological_mask(
    series: np.ndarray,
    original_mask: np.ndarray,
    strategy: str,
    target_ratio: float = 0.20,
    seed: int = 42,
    slope_thresh: float = 0.05,
    min_run: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create an evaluation mask using a physiological masking strategy.

    Masking is done in *whole contiguous blocks* so that baselines must bridge
    a realistic gap rather than fill isolated scattered points.  The total
    number of masked observed timesteps is controlled by target_ratio.

    Args:
        series        : (L, 1) normalized CGM values
        original_mask : (L, 1) binary observed mask  (1 = observed)
        strategy      : one of
                          "meal_post"  - post-prandial windows
                          "sleep"      - nocturnal sleep period
                          "ascending"  - detected glucose rises
                          "dipping"    - detected glucose drops
                          "combined"   - meal + ascending + dipping
        target_ratio  : approx. fraction of observed points to mask
        seed          : RNG seed for reproducibility
        slope_thresh  : per-timestep slope threshold for ascending/dipping
        min_run       : minimum run length for slope-based detection

    Returns:
        eval_mask   : (L, 1) - original_mask with masked positions zeroed out
                      (give this to the baselines as "what they see")
        target_mask : (L, 1) - 1 where the position was masked for evaluation
    """
    rng = np.random.RandomState(seed)
    L = series.shape[0]
    if L == 0:
        return original_mask.copy(), np.zeros_like(original_mask)
    obs = original_mask[:, 0].astype(bool)
    s1d = series[:, 0].astype(np.float32)

    n_obs = int(obs.sum())
    n_target = max(1, int(round(n_obs * target_ratio)))

    # - build physiological flag -
    if strategy == "meal_post":
        flag = flag_meal_timesteps(L)
    elif strategy == "sleep":
        flag = flag_sleep_timesteps(L)
    elif strategy == "ascending":
        flag = flag_ascending_segments(
            s1d, obs, slope_thresh=slope_thresh, min_run=min_run
        )
    elif strategy == "dipping":
        flag = flag_dipping_segments(
            s1d, obs, slope_thresh=slope_thresh, min_run=min_run
        )
    elif strategy == "combined":
        flag = (
            flag_meal_timesteps(L)
            | flag_ascending_segments(
                s1d, obs, slope_thresh=slope_thresh, min_run=min_run
            )
            | flag_dipping_segments(
                s1d, obs, slope_thresh=slope_thresh, min_run=min_run
            )
        )
    else:
        raise ValueError(f"Unknown physiological masking strategy: {strategy!r}")

    # - extract contiguous blocks within flagged + observed region -
    blocks = _extract_contiguous_blocks(flag, obs)
    n_flagged_obs = int((flag & obs).sum())

    if n_flagged_obs == 0 or len(blocks) == 0:
        # No physiologically-flagged observed points → fall back to a single
        # random contiguous block (same as the existing block-gap approach).
        obs_idx = np.where(obs)[0]
        if len(obs_idx) == 0:
            eval_mask = original_mask.copy()
            target_mask = np.zeros_like(original_mask)
            return eval_mask, target_mask
        block_len = min(n_target, len(obs_idx) - 1)
        start = rng.randint(0, max(1, len(obs_idx) - block_len))
        chosen_start = obs_idx[start]
        blocks = [(chosen_start, min(chosen_start + block_len, L))]

    # - select blocks greedily until ~n_target points are masked -
    masked_bool = _blocks_to_mask(blocks, L, obs, n_target, rng)

    eval_mask_1d = obs.astype(np.float32).copy()
    eval_mask_1d[masked_bool] = 0.0

    target_mask_1d = (obs.astype(np.float32) - eval_mask_1d).clip(0, 1)

    return (eval_mask_1d.reshape(-1, 1), target_mask_1d.reshape(-1, 1))


# Convenience: flag statistics for a dataset


def summarize_flags(
    samples: list[dict], strategies: list[str] | None = None, L: int | None = None
) -> dict:
    """
    Print and return per-strategy coverage statistics over a list of samples.

    Useful for sanity-checking that each strategy actually flags a meaningful
    fraction of the observed data before running the full experiment.
    L is inferred per sample when not provided.
    """
    if strategies is None:
        strategies = ["meal_post", "sleep", "ascending", "dipping", "combined"]

    stats: dict[str, dict] = {}
    for strat in strategies:
        flagged_ratios = []
        for s in samples:
            obs = s["irg_ts_mask"][:, 0].astype(bool)
            ser = s["irg_ts"][:, 0].astype(np.float32)
            if obs.sum() == 0:
                continue
            sample_L = L if L is not None else len(obs)
            if strat == "meal_post":
                flag = flag_meal_timesteps(sample_L)
            elif strat == "sleep":
                flag = flag_sleep_timesteps(sample_L)
            elif strat == "ascending":
                flag = flag_ascending_segments(ser, obs)
            elif strat == "dipping":
                flag = flag_dipping_segments(ser, obs)
            elif strat == "combined":
                flag = (
                    flag_meal_timesteps(sample_L)
                    | flag_ascending_segments(ser, obs)
                    | flag_dipping_segments(ser, obs)
                )
            else:
                continue
            ratio = float((flag & obs).sum()) / float(obs.sum())
            flagged_ratios.append(ratio)
        if flagged_ratios:
            stats[strat] = {
                "mean_flagged_ratio": float(np.mean(flagged_ratios)),
                "median_flagged_ratio": float(np.median(flagged_ratios)),
                "frac_with_any_flag": float(np.mean([r > 0 for r in flagged_ratios])),
                "n_samples": len(flagged_ratios),
            }
        else:
            stats[strat] = {"mean_flagged_ratio": 0.0, "n_samples": 0}

    for strat, st in stats.items():
        print(
            f"  {strat:12s}: mean_flagged={st['mean_flagged_ratio']:.2%}  "
            f"median={st.get('median_flagged_ratio', 0):.2%}  "
            f"any_flag={st.get('frac_with_any_flag', 0):.2%}  "
            f"n={st['n_samples']}"
        )
    return stats


if __name__ == "__main__":
    import pickle, sys

    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/aireadi_cgm_full/aireadi_cgm_test.pkl"
    )
    with open(path, "rb") as f:
        test = pickle.load(f)
    print(f"Loaded {len(test)} test samples.")
    print("Flag coverage statistics:")
    summarize_flags(test)
