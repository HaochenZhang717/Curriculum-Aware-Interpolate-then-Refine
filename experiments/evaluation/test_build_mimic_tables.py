import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SCRIPT = os.path.join(HERE, "build_mimic_tables.py")


def _cell(method, mech, pct, rmse, cu, mort):
    return {
        "method": method,
        "mech": mech,
        "pct": pct,
        "rmse": rmse,
        "bias": 0.0,
        "empse": 1.0,
        "mrr_tir": 0.5,
        "mrr_tar180": 0.4,
        "mrr_tbr70": 0.3,
        "mrr_mage": 0.2,
        "mrr_cv": 0.1,
        "gmi_abserr": 0.1,
        "mean_abserr": 0.1,
        "careunit": cu,
        "mortality": mort,
    }


def test_tables_emit_rmse_and_strata(tmp_path):
    cells = []
    for m in ("refine", "linear"):
        for mech in ("mcar", "mar", "nmar"):
            for pct in (0.1, 0.2):
                r = 5.0 if m == "refine" else 7.0
                cells += [
                    _cell(m, mech, pct, r, "MICU", 0),
                    _cell(m, mech, pct, r + 1, "SICU", 1),
                ]
    src = tmp_path / "toye_mimic_abp.json"
    json.dump({"dataset": "mimic_abp", "units": "mmHg", "cells": cells}, open(src, "w"))
    out = tmp_path / "tables"
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT}/baselines/statistical")
    subprocess.run(
        [sys.executable, SCRIPT, "--cells", str(src), "--outdir", str(out)],
        check=True,
        env=env,
    )
    rmse_md = (out / "mimic_abp_rmse.md").read_text()
    assert "refine" in rmse_md and "linear" in rmse_md and "nmar" in rmse_md.lower()
    strat_md = (out / "mimic_abp_strata.md").read_text()
    assert "MICU" in strat_md and "SICU" in strat_md
    assert (out / "mimic_abp_mrr.md").exists()
