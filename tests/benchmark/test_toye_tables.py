import json
import numpy as np
from experiments.evaluation.toye_tables import build_metric_table


def test_build_metric_table_pivots(tmp_path):
    d = {
        "aggregate": {
            "linear|mcar|0.1": {"rmse": 12.0, "mrr_tir": 0.9},
            "cair|mcar|0.1": {"rmse": 13.0, "mrr_tir": 0.95},
        }
    }
    p = tmp_path / "r.json"
    p.write_text(json.dumps(d))
    tbl = build_metric_table(str(p), metric="mrr_tir", mech="mcar")
    # rows = methods, cols = percentages
    assert tbl[("cair", 0.1)] == 0.95
    assert tbl[("linear", 0.1)] == 0.9
