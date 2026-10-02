import numpy as np
from experiments.evaluation.eval_toye_benchmark import INPROC_BASELINES, aggregate_cells


def test_baseline_registry_has_expected_methods():
    for name in [
        "linear",
        "locf",
        "mean",
        "mode",
        "knn",
        "hotdeck",
        "mice",
        "missforest",
        "fourier",
    ]:
        assert name in INPROC_BASELINES
        assert callable(INPROC_BASELINES[name])


def test_aggregate_cells_means_over_records():
    # two records, one metric 'rmse' for method 'linear' mechanism 'mcar' p=0.1
    cells = [
        {"method": "linear", "mech": "mcar", "pct": 0.1, "rmse": 10.0},
        {"method": "linear", "mech": "mcar", "pct": 0.1, "rmse": 20.0},
    ]
    agg = aggregate_cells(cells, value_keys=["rmse"])
    key = ("linear", "mcar", 0.1)
    assert abs(agg[key]["rmse"] - 15.0) < 1e-9
