"""Canonical REFINE publish-time configuration."""

from __future__ import annotations

BEST_MEMBER_KWARGS = {
    "interp_hidden": 128,
    "interp_layers": 4,
    "inject": "residual",
    "interp_type": "gru",
    "interp_in": "basic",
    "d_model": 128,
    "n_layers": 8,
    "ff_dim": 512,
    "n_heads": 8,
    "window": 576,
}

BEST_TRAINING_CONFIG = {
    "aux_loss_weight": 0.7,
    "aux_huber": False,
    "learning_rate": 3e-4,
    "warm_start": False,
    "ema": True,
    "loss_weights": (0.15, 0.35, 0.50),
    "member_epochs": {
        "member0_hpo3_a07_l4.pt": 120,
        "member1_hpo4_seed1.pt": 100,
        "member2_hpo4_seed2.pt": 100,
        "member3_hpo4_seed3.pt": 100,
        "member4_hpo4_seed4.pt": 100,
    },
}

BEST_INFERENCE_CONFIG = {
    "stride": 144,
    "n_refinements": 2,
}

BEST_ENSEMBLE_MEMBERS = (
    "member0_hpo3_a07_l4.pt",
    "member1_hpo4_seed1.pt",
    "member2_hpo4_seed2.pt",
    "member3_hpo4_seed3.pt",
    "member4_hpo4_seed4.pt",
)

BEST_EVAL_SUMMARY = {
    "dataset": "aireadi",
    "protocol": "352 test participants (full test set) x 5 masks x 5 strategies, day-2 [288:576]",
    "expected_avg_rmse_mgdl": 14.61,  # 5-seed ensemble, full test set
}

__all__ = [
    "BEST_MEMBER_KWARGS",
    "BEST_TRAINING_CONFIG",
    "BEST_INFERENCE_CONFIG",
    "BEST_ENSEMBLE_MEMBERS",
    "BEST_EVAL_SUMMARY",
]
