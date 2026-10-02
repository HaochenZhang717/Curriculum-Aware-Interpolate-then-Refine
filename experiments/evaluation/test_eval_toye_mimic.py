from utils.paths import DATA_ROOT, REPO_ROOT, CACHE_ROOT
import json, os, pickle, subprocess, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SCRIPT = os.path.join(HERE, "eval_toye_mimic.py")
PYPOTS_PY = os.environ.get("PYPOTS_PYTHON", __import__("sys").executable)


def _rec(subj, T=288, cov=0.9):
    rng = np.random.default_rng(subj)
    z = 0.5 * np.sin(2 * np.pi * np.arange(T) / T) + 0.2 * rng.standard_normal(T)
    m = (rng.random(T) < cov).astype(np.float32)
    out = {
        "person_id": subj,
        "subject_id": subj,
        "irg_ts": z.reshape(T, 1).astype(np.float32),
        "irg_ts_mask": m.reshape(T, 1),
        "ts_tt": np.arange(T, dtype=np.float32),
        "seq_len": T,
        "dataset_name": "mimic_abp",
        "sampling_minutes": 5,
        "careunit": "MICU" if subj % 2 else "SICU",
        "in_hospital_mortality": subj % 2,
    }
    for slot in ("hr", "resp", "spo2"):
        out[f"mm_{slot}"] = (
            (0.3 * rng.standard_normal(T)).reshape(T, 1).astype(np.float32)
        )
        out[f"mm_{slot}_p"] = m.reshape(T, 1)
    return out


def test_eval_toye_mimic_runs_baselines(tmp_path):
    d = tmp_path / "mimic_abp"
    d.mkdir()
    for sp in ("train", "val", "test"):
        pickle.dump([_rec(i) for i in range(4)], open(d / f"{sp}.pkl", "wb"))
    meta = {
        "name": "mimic_abp",
        "display_name": "MIMIC ABP",
        "sampling_minutes": 5,
        "day_len": 288,
        "eval_window_start": 0,
        "eval_window_end": 288,
        "glucose_mean_mgdl": 80.0,
        "glucose_std_mgdl": 12.0,
        "glucose_normalization": "native_abp_mean_zscore",
        "aligned_to_midnight": False,
        "notes": {
            "nmar_lo": 65.0,
            "nmar_hi": 100.0,
            "mar_covariate": "hr",
            "burden": {
                "normal_lo": 70.0,
                "normal_hi": 100.0,
                "low_thr": 65.0,
                "high_thr": 100.0,
            },
        },
    }
    json.dump(meta, open(d / "metadata.json", "w"))
    out = tmp_path / "toye_mimic.json"
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT}/baselines/statistical")
    subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--dataset",
            "mimic_abp",
            "--data_dir",
            str(d),
            "--gpu",
            "-1",
            "--n_participants",
            "4",
            "--skip_cair",
            "--out",
            str(out),
        ],
        check=True,
        env=env,
    )
    res = json.load(open(out))
    methods = {c["method"] for c in res["cells"]}
    assert "linear" in methods and "mice" in methods
    c = res["cells"][0]
    for k in (
        "rmse",
        "bias",
        "empse",
        "mrr_tir",
        "careunit",
        "mortality",
        "mech",
        "pct",
    ):
        assert k in c
    assert set(m for m in ("mcar", "mar", "nmar")) & {c["mech"] for c in res["cells"]}


def _write_fixture(tmp_path):
    d = tmp_path / "mimic_abp"
    d.mkdir()
    for sp in ("train", "val", "test"):
        pickle.dump([_rec(i) for i in range(4)], open(d / f"{sp}.pkl", "wb"))
    meta = {
        "name": "mimic_abp",
        "display_name": "MIMIC ABP",
        "sampling_minutes": 5,
        "day_len": 288,
        "eval_window_start": 0,
        "eval_window_end": 288,
        "glucose_mean_mgdl": 80.0,
        "glucose_std_mgdl": 12.0,
        "glucose_normalization": "native_abp_mean_zscore",
        "aligned_to_midnight": False,
        "notes": {
            "nmar_lo": 65.0,
            "nmar_hi": 100.0,
            "mar_covariate": "hr",
            "burden": {
                "normal_lo": 70.0,
                "normal_hi": 100.0,
                "low_thr": 65.0,
                "high_thr": 100.0,
            },
        },
    }
    json.dump(meta, open(d / "metadata.json", "w"))
    return d


@pytest.mark.skipif(
    not os.path.exists(PYPOTS_PY), reason="PyPOTS env not available on this host"
)
def test_eval_toye_mimic_runs_pypots(tmp_path):
    d = _write_fixture(tmp_path)
    out = tmp_path / "toye_mimic_pypots.json"
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT}/baselines/statistical")
    subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--dataset",
            "mimic_abp",
            "--data_dir",
            str(d),
            "--gpu",
            "-1",
            "--n_participants",
            "4",
            "--skip_cair",
            "--pypots_python",
            PYPOTS_PY,
            "--pypots_epochs",
            "2",
            "--out",
            str(out),
        ],
        check=True,
        env=env,
    )
    res = json.load(open(out))
    methods = {c["method"] for c in res["cells"]}
    assert "mrnn" in methods and "gpvae" in methods
    for m in ("mrnn", "gpvae"):
        rmses = [c["rmse"] for c in res["cells"] if c["method"] == m]
        assert rmses and all(np.isfinite(r) for r in rmses)
