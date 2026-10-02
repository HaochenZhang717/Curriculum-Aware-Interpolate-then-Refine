import numpy as np
from utils.missingness_mechanisms import create_mcar_mask


def test_mcar_deletes_requested_fraction(synthetic_day):
    series, mask = synthetic_day
    p = 0.20
    eval_mask, target_mask = create_mcar_mask(series, mask, percent=p, seed=0)
    n_obs = int(mask[:, 0].sum())
    n_del = int(target_mask[:, 0].sum())
    # within +/- 3% of requested
    assert abs(n_del / n_obs - p) < 0.03
    tgt = target_mask[:, 0].astype(bool)
    assert (eval_mask[:, 0][tgt] == 0).all()
    # never deletes an already-missing point
    assert (target_mask[:, 0][mask[:, 0] == 0] == 0).all()


def test_mcar_deterministic(synthetic_day):
    series, mask = synthetic_day
    a = create_mcar_mask(series, mask, percent=0.15, seed=7)[1]
    b = create_mcar_mask(series, mask, percent=0.15, seed=7)[1]
    assert np.array_equal(a, b)
