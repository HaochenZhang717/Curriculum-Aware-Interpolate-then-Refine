import numpy as np
from baselines.toye.toye_baselines import impute_fourier


def test_fourier_recovers_periodic_signal():
    L = 288
    t = np.arange(L)
    s = np.sin(2 * np.pi * t / 48).astype(np.float32)  # strongly periodic
    m = np.ones(L, dtype=np.float32)
    m[100:130] = 0.0
    s_seen = s.copy()
    s_seen[m == 0] = np.nan
    out = impute_fourier(s_seen, m, n_freqs=6, n_iters=50)
    err = np.sqrt(np.mean((out[100:130] - s[100:130]) ** 2))
    assert err < 0.2  # periodic signal recovered well
    obs = m.astype(bool)
    assert np.allclose(out[obs], s[obs], atol=1e-3)
