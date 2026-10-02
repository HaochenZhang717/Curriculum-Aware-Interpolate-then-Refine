import numpy as np
import pytest
from experiments.evaluation.toye_downstream import mrr_from_traces, load_battery


def test_load_battery_available():
    # If the battery module is present, it imports; otherwise skip cleanly.
    try:
        b = load_battery()
    except FileNotFoundError:
        pytest.skip("downstream battery not present in this checkout")
    assert callable(b)


def test_mrr_perfect_reconstruction_is_one():
    try:
        load_battery()
    except FileNotFoundError:
        pytest.skip("downstream battery not present")
    L = 288
    t = np.arange(L)
    true = (120 + 40 * np.sin(2 * np.pi * t / 288)).astype(np.float32)
    meanfill = np.full(L, float(true.mean()), dtype=np.float32)
    # perfect method == true -> err(method)=0 -> MRR=1 for every metric
    mrr = mrr_from_traces(
        true_mgdl=true, method_mgdl=true.copy(), meanfill_mgdl=meanfill
    )
    assert all(abs(v - 1.0) < 1e-6 for v in mrr.values() if v == v)
