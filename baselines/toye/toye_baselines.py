#!/usr/bin/env python3
"""Toye et al. (CHIL 2025) imputation baselines with a uniform interface."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _seen(series1d, obs_mask1d):
    """Series with missing positions set to NaN, plus boolean observed mask."""
    obs = np.asarray(obs_mask1d).astype(bool)
    s = np.asarray(series1d, dtype=np.float64).copy()
    s[~obs] = np.nan
    return s, obs


def impute_linear(series1d, obs_mask1d):
    s, obs = _seen(series1d, obs_mask1d)
    idx = np.arange(len(s))
    if obs.sum() == 0:
        return np.zeros_like(s)
    out = np.interp(idx, idx[obs], s[obs])
    return out.astype(np.float32)


def impute_locf(series1d, obs_mask1d):
    s, obs = _seen(series1d, obs_mask1d)
    ser = pd.Series(s).ffill().bfill()
    return ser.to_numpy(dtype=np.float32)


def impute_mean(series1d, obs_mask1d):
    s, obs = _seen(series1d, obs_mask1d)
    fill = np.nanmean(s) if obs.any() else 0.0
    out = np.where(obs, np.nan_to_num(s), fill)
    return out.astype(np.float32)


def impute_mode(series1d, obs_mask1d):
    s, obs = _seen(series1d, obs_mask1d)
    if obs.any():
        vals, counts = np.unique(np.round(s[obs], 3), return_counts=True)
        fill = float(vals[np.argmax(counts)])
    else:
        fill = 0.0
    out = np.where(obs, np.nan_to_num(s), fill)
    return out.astype(np.float32)


def _hankel(s, w):
    """Overlapping windows of width w -> (L, w) with NaNs preserved."""
    L = len(s)
    M = np.full((L, w), np.nan, dtype=np.float64)
    for j in range(w):
        M[: L - j, j] = s[j:]
    return M


def _invert_hankel(M, L, w):
    """Average the overlapping predictions back to a (L,) series."""
    acc = np.zeros(L)
    cnt = np.zeros(L)
    for j in range(w):
        col = M[: L - j, j]
        acc[j:] += col
        cnt[j:] += 1.0
    cnt[cnt == 0] = 1.0
    return (acc / cnt).astype(np.float32)


def _preserve_obs(out, series1d, obs_mask1d):
    obs = np.asarray(obs_mask1d).astype(bool)
    out = np.asarray(out, dtype=np.float32).copy()
    out[obs] = np.asarray(series1d, dtype=np.float32)[obs]
    return out


def _tabular_impute(series1d, obs_mask1d, imputer, w=12):
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    s, obs = _seen(series1d, obs_mask1d)
    L = len(s)
    if obs.sum() < 2:
        return impute_mean(series1d, obs_mask1d)
    M = _hankel(s, w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        Mf = imputer.fit_transform(M)
    out = _invert_hankel(Mf, L, w)
    return _preserve_obs(out, series1d, obs_mask1d)


def impute_knn(series1d, obs_mask1d, k=5):
    from sklearn.impute import KNNImputer

    return _tabular_impute(series1d, obs_mask1d, KNNImputer(n_neighbors=k))


def impute_hotdeck(series1d, obs_mask1d):
    from sklearn.impute import KNNImputer

    return _tabular_impute(series1d, obs_mask1d, KNNImputer(n_neighbors=1))


def impute_mice(series1d, obs_mask1d):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    return _tabular_impute(
        series1d, obs_mask1d, IterativeImputer(max_iter=10, random_state=0)
    )


def impute_missforest(series1d, obs_mask1d):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import RandomForestRegressor

    est = RandomForestRegressor(n_estimators=50, random_state=0, n_jobs=1)
    return _tabular_impute(
        series1d,
        obs_mask1d,
        IterativeImputer(estimator=est, max_iter=5, random_state=0),
    )


def impute_fourier(series1d, obs_mask1d, n_freqs=8, n_iters=100):
    """Iterative Fourier reconstruction (Rahman et al., 2015).

    Start from a linear fill, then repeatedly: FFT, keep the n_freqs dominant
    components, inverse FFT, and re-impose observed values. Converges to a
    periodic reconstruction that respects the observed points."""
    s, obs = _seen(series1d, obs_mask1d)
    L = len(s)
    if obs.sum() < 2:
        return impute_mean(series1d, obs_mask1d)
    est = impute_linear(series1d, obs_mask1d).astype(np.float64)
    obs_vals = np.asarray(series1d, dtype=np.float64)
    for _ in range(n_iters):
        F = np.fft.rfft(est)
        mag = np.abs(F)
        if len(mag) > n_freqs + 1:
            keep = np.argsort(mag)[-(n_freqs + 1) :]
            Fk = np.zeros_like(F)
            Fk[keep] = F[keep]
        else:
            Fk = F
        rec = np.fft.irfft(Fk, n=L)
        rec[obs] = obs_vals[obs]  # re-impose observed
        if np.allclose(rec, est, atol=1e-6):
            est = rec
            break
        est = rec
    return _preserve_obs(est, series1d, obs_mask1d)
