import numpy as np
from utils.physiological_masking import create_physiological_mask


def test_existing_mask_contract(synthetic_day):
    series, mask = synthetic_day
    eval_mask, target_mask = create_physiological_mask(
        series, mask, strategy="combined", target_ratio=0.20, seed=1
    )
    assert eval_mask.shape == series.shape
    assert target_mask.shape == series.shape
    # deleted positions are zeroed in eval_mask and flagged in target_mask
    tgt = target_mask[:, 0].astype(bool)
    assert tgt.sum() > 0
    assert (eval_mask[:, 0][tgt] == 0).all()
