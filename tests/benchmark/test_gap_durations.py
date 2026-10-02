import numpy as np
from utils.missingness_mechanisms import native_gap_durations


def test_native_gap_durations_counts_runs():
    # mask over (L,1); zeros are native-missing runs of length 3 and 2
    L = 12
    m = np.ones((L, 1), dtype=np.float32)
    m[2:5, 0] = 0.0  # run length 3
    m[8:10, 0] = 0.0  # run length 2
    durs = native_gap_durations(m)
    assert sorted(durs) == [2, 3]


def test_native_gap_durations_empty_when_all_observed():
    m = np.ones((10, 1), dtype=np.float32)
    assert native_gap_durations(m) == []
