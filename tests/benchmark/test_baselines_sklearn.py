import numpy as np
from baselines.toye.toye_baselines import (
    impute_knn,
    impute_hotdeck,
    impute_mice,
    impute_missforest,
)


def _gapped():
    L = 60
    t = np.arange(L)
    s = (0.5 * np.sin(2 * np.pi * t / 24)).astype(np.float32)
    m = np.ones(L, dtype=np.float32)
    m[20:26] = 0.0
    s_seen = s.copy()
    s_seen[m == 0] = np.nan
    return s, s_seen, m


import pytest


@pytest.mark.parametrize(
    "fn", [impute_knn, impute_hotdeck, impute_mice, impute_missforest]
)
def test_returns_full_length_no_nan(fn):
    s, s_seen, m = _gapped()
    out = fn(s_seen, m)
    assert out.shape == s.shape
    assert not np.isnan(out).any()
    # observed positions are preserved
    obs = m.astype(bool)
    assert np.allclose(out[obs], s[obs], atol=1e-3)
