import numpy as np
from utils.missingness_mechanisms import create_nmar_mask


def test_nmar_targets_extreme_glucose():
    # build a series in mg/dL then normalize; a hypo dip at [50:70]
    L = 288
    mean, std = 132.05, 42.33
    g = np.full(L, 120.0, dtype=np.float32)
    g[50:70] = 55.0  # below 70 -> NMAR trigger
    z = ((g - mean) / std).astype(np.float32).reshape(L, 1)
    mask = np.ones((L, 1), dtype=np.float32)
    _, target_mask = create_nmar_mask(
        z,
        mask,
        percent=0.08,
        gap_durations=[6],
        seed=2,
        mean_mgdl=mean,
        std_mgdl=std,
        lo_mgdl=70.0,
        hi_mgdl=150.0,
    )
    tgt = target_mask[:, 0].astype(bool)
    assert tgt.sum() > 0
    # deletions concentrate in the hypo region
    assert tgt[45:75].sum() / tgt.sum() > 0.6
