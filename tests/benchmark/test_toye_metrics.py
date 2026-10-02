import numpy as np
from experiments.evaluation.toye_metrics import rmse_mgdl, bias_mgdl, empse_mgdl


def test_metrics_on_known_values():
    std = 42.33
    # 3 target points; pred - true = +1, -1, +1 in z-space
    true = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pred = np.array([1.0, -1.0, 1.0], dtype=np.float32)
    tgt = np.array([True, True, True])
    # RMSE = sqrt(mean(1,1,1)) * std = std
    assert abs(rmse_mgdl(pred, true, tgt, std) - std) < 1e-4
    # bias = mean(1,-1,1) * std = std/3
    assert abs(bias_mgdl(pred, true, tgt, std) - std / 3.0) < 1e-3
    # EmpSE = std of predictions (ddof=1) * std_mgdl
    exp = np.std(pred, ddof=1) * std
    assert abs(empse_mgdl(pred, true, tgt, std) - exp) < 1e-3


def test_metrics_empty_target_is_nan():
    tgt = np.array([False, False])
    z = np.zeros(2, dtype=np.float32)
    assert np.isnan(rmse_mgdl(z, z, tgt, 42.33))
