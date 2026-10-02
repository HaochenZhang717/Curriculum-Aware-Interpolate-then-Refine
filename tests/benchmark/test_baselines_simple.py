import numpy as np
from baselines.toye.toye_baselines import (
    impute_linear,
    impute_locf,
    impute_mean,
    impute_mode,
)


def _obs_series():
    L = 20
    s = np.linspace(0.0, 1.0, L).astype(np.float32)
    m = np.ones(L, dtype=np.float32)
    m[5:10] = 0.0
    return s, m


def test_linear_bridges_gap_linearly():
    s, m = _obs_series()
    s_seen = s.copy()
    s_seen[m == 0] = np.nan
    out = impute_linear(s_seen, m)
    # linear ground truth: series is itself linear, so recovery is near-exact
    assert np.allclose(out, s, atol=1e-4)


def test_locf_carries_forward():
    s, m = _obs_series()
    s_seen = s.copy()
    s_seen[m == 0] = np.nan
    out = impute_locf(s_seen, m)
    assert np.allclose(out[5:10], s[4])


def test_mean_fills_with_observed_mean():
    s, m = _obs_series()
    s_seen = s.copy()
    s_seen[m == 0] = np.nan
    out = impute_mean(s_seen, m)
    assert np.allclose(out[5:10], np.nanmean(s_seen))


def test_mode_returns_full_length():
    s, m = _obs_series()
    s_seen = s.copy()
    s_seen[m == 0] = np.nan
    out = impute_mode(s_seen, m)
    assert out.shape == s.shape
    assert not np.isnan(out).any()
