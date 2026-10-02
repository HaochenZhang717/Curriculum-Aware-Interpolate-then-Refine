import numpy as np
from utils.missingness_mechanisms import create_mar_mask


def test_mar_targets_high_trigger_regions(synthetic_day):
    series, mask = synthetic_day
    L = series.shape[0]
    # trigger is high in one region [100:140], low elsewhere
    trigger = np.zeros(L, dtype=np.float32)
    trigger[100:140] = 10.0
    eval_mask, target_mask = create_mar_mask(
        series, mask, trigger=trigger, percent=0.10, gap_durations=[6], seed=3
    )
    tgt = target_mask[:, 0].astype(bool)
    assert tgt.sum() > 0
    # majority of deletions fall inside the high-trigger region
    frac_in = tgt[100:140].sum() / tgt.sum()
    assert frac_in > 0.6
    assert (eval_mask[:, 0][tgt] == 0).all()


def test_mar_respects_percent_budget(synthetic_day):
    series, mask = synthetic_day
    L = series.shape[0]
    trigger = np.random.RandomState(0).rand(L).astype(np.float32)
    _, target_mask = create_mar_mask(
        series, mask, trigger=trigger, percent=0.15, gap_durations=[6], seed=1
    )
    n_obs = int(mask[:, 0].sum())
    # deleted count does not exceed budget by more than one window (6)
    assert target_mask[:, 0].sum() <= round(n_obs * 0.15) + 6
