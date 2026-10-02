import json, os, pickle, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "make_target_split.py")


def _fake_record(T, hr_cov):
    rng = np.random.default_rng(0)
    p = np.zeros((T, 1), np.float32)
    p[: int(T * hr_cov)] = 1.0
    return {
        "person_id": 1,
        "irg_ts": rng.standard_normal((T, 1)).astype(
            np.float32
        ),  # CGM (to be overwritten)
        "irg_ts_mask": np.ones((T, 1), np.float32),
        "ts_tt": np.arange(T, dtype=np.float32),
        "seq_len": T,
        "mm_hr": rng.standard_normal((T, 1)).astype(np.float32),
        "mm_hr_p": p,
        "mm_resp": rng.standard_normal((T, 1)).astype(np.float32),
        "mm_resp_p": np.ones((T, 1), np.float32),
    }


def test_builder_swaps_target_and_drops_zero_coverage(tmp_path):
    mm = tmp_path / "mm"
    mm.mkdir()
    r_ok = _fake_record(600, hr_cov=0.8)
    r_zero = _fake_record(600, hr_cov=0.0)
    for sp in ("train", "val", "test"):
        pickle.dump([r_ok, r_zero], open(mm / f"aireadi_cgm_{sp}.pkl", "wb"))
    json.dump(
        {
            "mm_stats": {
                "hr": {"mean": 75.944, "std": 14.968},
                "resp": {"mean": 10.352, "std": 6.559},
            }
        },
        open(mm / "mm_norm_stats.json", "w"),
    )

    out = tmp_path / "hr"
    subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--mm_dir",
            str(mm),
            "--target",
            "hr",
            "--out",
            str(out),
        ],
        check=True,
    )

    recs = pickle.load(open(out / "aireadi_cgm_test.pkl", "rb"))
    assert len(recs) == 1  # zero-coverage record dropped
    np.testing.assert_array_equal(recs[0]["irg_ts"], r_ok["mm_hr"])  # target swapped in
    np.testing.assert_array_equal(recs[0]["irg_ts_mask"], r_ok["mm_hr_p"])
    meta = json.load(open(out / "metadata.json"))
    assert meta["name"] == "aireadi"
    assert (
        abs(meta["glucose_std_mgdl"] - 14.968) < 1e-6
    )  # native std => native-unit RMSE
