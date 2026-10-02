"""CPU-only tests for the MIMIC clinical-downstream imputation-recovery eval."""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (ROOT, os.path.join(ROOT, "baselines", "statistical")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.evaluation import mimic_clinical_downstream as mcd  # noqa: E402


def _make_records(n=30, T=288, seed=0):
    """Synthetic records whose mean drives a synthetic label (AUROC>0.5 reachable).

    Each record carries z-scored irg_ts/irg_ts_mask and a stay_id. The label is
    1 when the native mean exceeds the population median, so the mean/slope
    features carry real signal that the reference classifier can learn.
    """
    rng = np.random.RandomState(seed)
    mean, std = 80.0, 15.0
    recs = []
    stay_labels = {}
    native_means = []
    for i in range(n):
        base = rng.uniform(60, 105)  # native-unit level
        native = base + rng.randn(T) * 4.0  # native trace
        z = (native - mean) / std
        mask = (rng.rand(T) > 0.15).astype(np.float32)  # ~85% observed
        mask[0] = 1.0
        mask[-1] = 1.0
        sid = 100000 + i
        recs.append(
            {
                "irg_ts": z.reshape(T, 1).astype(np.float32),
                "irg_ts_mask": mask.reshape(T, 1).astype(np.float32),
                "stay_id": sid,
            }
        )
        native_means.append(base)
    thr = float(np.median(native_means))
    for i, rec in enumerate(recs):
        sid = int(rec["stay_id"])
        pos = int(native_means[i] > thr)
        # mortality is the signal-bearing label; sepsis/hf get some spread too.
        stay_labels[sid] = {
            "mortality": pos,
            "sepsis": int(native_means[i] > thr + 5),
            "hf": int(rng.rand() > 0.5),
        }
    return recs, stay_labels, mean, std


def test_derive_labels_hand_built(tmp_path):
    """Label derivation picks sepsis / hf / mortality from a hand-built stay."""
    import pickle

    stays = {
        1: {
            "stay_id": 1,
            "subject_id": 10,
            "labels": {"in_hospital_mortality": 1},
            "diagnoses": [
                (1, "038.9", 9, "Unspecified septicemia"),
                (2, "4280", 9, "Congestive heart failure"),
            ],
        },
        2: {
            "stay_id": 2,
            "subject_id": 11,
            "labels": {"in_hospital_mortality": 0},
            "diagnoses": [(1, "4241", 9, "Aortic valve disorders")],
        },
        3: {
            "stay_id": 3,
            "subject_id": 12,
            "labels": {"in_hospital_mortality": 0},
            "diagnoses": [(1, "99592", 9, "Severe sepsis")],
        },
    }
    p = tmp_path / "src.pkl"
    with open(p, "wb") as f:
        pickle.dump({"stays": stays}, f)
    labels = mcd.derive_labels(str(p))

    assert labels[1] == {"mortality": 1, "sepsis": 1, "hf": 1}  # 0389 + 428x
    assert labels[2] == {"mortality": 0, "sepsis": 0, "hf": 0}  # valve only
    assert labels[3] == {"mortality": 0, "sepsis": 1, "hf": 0}  # 99592 sepsis


def test_features_shape():
    x = np.linspace(60, 100, 288).astype(np.float32)
    f = mcd.trace_features(x, 65.0, 100.0)
    assert f.shape == (14,)
    assert np.isfinite(f).all()


def test_complete_fills_native_gaps():
    T = 20
    trace = np.arange(T, dtype=np.float32) + 100.0
    obs = np.ones(T, dtype=np.float32)
    obs[5:9] = 0.0
    full = mcd.complete(trace, obs)
    # linear over a straight line reproduces the true values exactly
    assert np.allclose(full, trace, atol=1e-4)


def test_eval_real_ge_mean_auroc():
    """Oracle 'real' fill AUROC >= 'mean' fill AUROC on a signal-bearing label."""
    # >= 20 positives are required by the reference-training guard, so we build
    # enough synthetic records to clear that bar while staying CPU-fast.
    recs, labels, mean, std = _make_records(n=120, seed=1)
    n_train = 90
    train_recs = recs[:n_train]
    test_recs = recs[n_train:]

    models = mcd.train_reference(
        train_recs, labels, mean, std, thr_lo=65.0, thr_hi=100.0, n_seeds=3
    )
    assert "mortality" in models, "mortality label should be trainable"

    results = mcd.evaluate(
        test_recs,
        labels,
        models,
        mean,
        std,
        thr_lo=65.0,
        thr_hi=100.0,
        methods=["real", "mean", "linear"],
        mechs=("mcar",),
        rates=(0.2,),
        seed_base=7000,
    )

    def auroc(method):
        hit = [
            r
            for r in results
            if r["label"] == "mortality"
            and r["method"] == method
            and r["mech"] == "mcar"
            and r["rate"] == 0.2
        ]
        assert hit, f"missing result for {method}"
        return hit[0]["auroc"]

    a_real = auroc("real")
    a_mean = auroc("mean")
    assert a_real is not None and a_mean is not None
    # oracle fill should be at least as informative as the flat mean fill.
    assert a_real >= a_mean - 1e-9, f"real {a_real} < mean {a_mean}"


def test_output_json_schema(tmp_path, monkeypatch):
    """The run() output dict has the expected keys and row schema (CPU, no refine)."""
    import pickle

    recs, labels, mean, std = _make_records(n=120, seed=2)

    # source pickle so derive_labels() has a file to read; use the same stay_ids.
    stays = {}
    for rec in recs:
        sid = int(rec["stay_id"])
        lab = labels[sid]
        diags = []
        if lab["sepsis"]:
            diags.append((1, "0389", 9, "sepsis"))
        if lab["hf"]:
            diags.append((2, "4280", 9, "hf"))
        stays[sid] = {
            "stay_id": sid,
            "subject_id": sid,
            "labels": {"in_hospital_mortality": lab["mortality"]},
            "diagnoses": diags,
        }
    src = tmp_path / "src.pkl"
    with open(src, "wb") as f:
        pickle.dump({"stays": stays}, f)

    # stub the dataset loader so run() consumes our synthetic split.
    class _Meta:
        name = "mimic_synth"
        glucose_mean_mgdl = mean
        glucose_std_mgdl = std

    monkeypatch.setattr(
        mcd,
        "load_dataset_splits",
        lambda name: (recs[:90], recs[:90], recs[90:], _Meta()),
    )

    out_path = tmp_path / "out.json"

    class _Args:
        dataset = "mimic_synth"
        source_pkl = str(src)
        device = "cpu"
        n_test = 30
        n_seeds = 2
        methods = "real,mean,linear"
        out = str(out_path)

    out = mcd.run(_Args())
    assert set(out.keys()) == {"dataset", "labels", "results", "bounds_note"}
    assert out["dataset"] == "mimic_synth"
    assert out["bounds_note"] == "real=oracle upper, mean=lower"
    assert isinstance(out["results"], list) and len(out["results"]) > 0
    for row in out["results"]:
        assert set(row.keys()) == {"label", "method", "mech", "rate", "auroc", "n"}
    assert os.path.exists(out_path)
