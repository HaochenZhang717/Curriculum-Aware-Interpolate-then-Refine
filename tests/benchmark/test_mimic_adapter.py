# tests/benchmark/test_mimic_adapter.py
import os, pickle
import numpy as np


def _fake_bundle(path, n_subjects=6, windows_per_subject=2, T=288):
    rng = np.random.default_rng(0)
    stays = {}
    wid = 0
    for subj in range(n_subjects):
        for _ in range(windows_per_subject):

            def chan(base, spread, cov):
                v = (base + spread * rng.standard_normal(T)).astype(np.float32)
                m = (rng.random(T) < cov).astype(np.float32)
                return v, m

            abp_m_v, abp_m_m = chan(80.0, 10.0, 0.85)
            abp = np.stack([abp_m_v + 40, abp_m_v - 20, abp_m_v], axis=1).astype(
                np.float32
            )
            abp_mask = np.repeat(abp_m_m.reshape(T, 1), 3, axis=1).astype(np.float32)
            hr_v, hr_m = chan(75.0, 12.0, 0.93)
            rr_v, rr_m = chan(18.0, 3.0, 0.92)
            sp_v, sp_m = chan(97.0, 1.5, 0.90)
            stays[wid] = {
                "stay_id": wid,
                "subject_id": subj,
                "hadm_id": subj * 10,
                "timeseries": {
                    "abp": abp,
                    "abp_mask": abp_mask,
                    "hr": hr_v.reshape(T, 1),
                    "hr_mask": hr_m.reshape(T, 1),
                    "rr": rr_v.reshape(T, 1),
                    "rr_mask": rr_m.reshape(T, 1),
                    "spo2": sp_v.reshape(T, 1),
                    "spo2_mask": sp_m.reshape(T, 1),
                },
                "metadata": {
                    "first_careunit": "MICU" if subj % 2 else "SICU",
                    "anchor_age": 60,
                    "gender": "M",
                },
                "labels": {
                    "in_hospital_mortality": subj % 2,
                    "icu_mortality": 0,
                    "los_days": 3.5,
                },
                "window_index": 0,
            }
            wid += 1
    pickle.dump({"config": {}, "registries": {}, "stays": stays}, open(path, "wb"))


def test_adapter_builds_abp_records(tmp_path):
    from cgm_datasets.mimic_adapter import MimicAbpAdapter

    raw = tmp_path / "mimic.pkl"
    _fake_bundle(str(raw))
    ad = MimicAbpAdapter(str(tmp_path))
    res = ad._prepare_raw(str(raw))
    rec = res.train[0]
    assert rec["irg_ts"].shape == (288, 1) and rec["irg_ts_mask"].shape == (288, 1)
    assert rec["dataset_name"] == "mimic_abp"
    for k in ("mm_hr", "mm_hr_p", "mm_resp", "mm_resp_p", "mm_spo2", "mm_spo2_p"):
        assert k in rec and np.asarray(rec[k]).shape == (288, 1)
    assert "careunit" in rec and "in_hospital_mortality" in rec and "subject_id" in rec
    assert 50.0 < res.metadata.glucose_mean_mgdl < 120.0
    assert res.metadata.glucose_std_mgdl > 0.0
    assert res.metadata.eval_window == (0, 288)


def test_adapter_splits_are_subject_disjoint(tmp_path):
    from cgm_datasets.mimic_adapter import MimicAbpAdapter

    raw = tmp_path / "mimic.pkl"
    _fake_bundle(str(raw))
    ad = MimicAbpAdapter(str(tmp_path))
    res = ad._prepare_raw(str(raw))
    subs = lambda recs: {int(r["subject_id"]) for r in recs}
    tr, va, te = subs(res.train), subs(res.val), subs(res.test)
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    test_subjects = [int(r["subject_id"]) for r in res.test]
    assert len(test_subjects) == len(set(test_subjects))


def test_hr_target_conditions_on_abp(tmp_path):
    from cgm_datasets.mimic_adapter import MimicHrAdapter

    raw = tmp_path / "mimic.pkl"
    _fake_bundle(str(raw))
    res = MimicHrAdapter(str(tmp_path))._prepare_raw(str(raw))
    rec = res.train[0]
    assert 40.0 < res.metadata.glucose_mean_mgdl < 110.0
    for k in ("mm_abp", "mm_abp_p", "mm_resp", "mm_spo2"):
        assert k in rec
