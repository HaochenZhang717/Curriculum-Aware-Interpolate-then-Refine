"""Adapters that read the nih_generation MIMIC pickles into REFINE's record schema."""

from __future__ import annotations

from utils.paths import DATA_ROOT, REPO_ROOT, CACHE_ROOT

import os
import pickle
from typing import Any

import numpy as np

from cgm_datasets.adapters import AdapterResult, BaseDatasetAdapter
from cgm_datasets.common import DatasetMetadata, standardize_record, summarize_splits
from cgm_datasets.mimic_profiles import PROFILES, VITAL_SOURCE, VITAL_TO_MM


def _subject_split(
    subject_ids: list[int],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
):
    subs = sorted({int(s) for s in subject_ids})
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    n = len(subs)
    if n < 3:
        raise ValueError(f"need at least 3 subjects to split, got {n}")
    # guarantee a nonempty val and test even for small cohorts (floor would zero them)
    n_va = max(1, int(n * ratios[1]))
    n_te = max(1, int(n * ratios[2]))
    n_tr = n - n_va - n_te
    return (set(subs[:n_tr]), set(subs[n_tr : n_tr + n_va]), set(subs[n_tr + n_va :]))


def _chan(ts: dict, key: str, idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (value (T,), mask (T,)) for source channel key at column idx."""
    v = np.asarray(ts[key], dtype=np.float32)
    m = np.asarray(ts[key + "_mask"], dtype=np.float32)
    if v.ndim == 2:
        v = v[:, idx]
    if m.ndim == 2:
        m = m[:, idx]
    return v.reshape(-1), m.reshape(-1)


def _zscore(v: np.ndarray, m: np.ndarray, mean: float, std: float) -> np.ndarray:
    std = std if std > 1e-6 else 1.0
    z = ((v - mean) / std).astype(np.float32)
    z[m <= 0] = 0.0  # zero where unobserved (masked out downstream)
    return z


class _MimicPickleAdapter(BaseDatasetAdapter):
    """Base for MIMIC vital-sign imputation adapters. Subclasses set target_key,
    default_raw, and (for hourly IV) _day_len / _sampling_minutes."""

    target_key: str = "abp_mean"
    default_raw: str = ""
    _day_len: int = 288
    _sampling_minutes: int = 5
    _max_test_windows_per_subject: int = 1

    def default_prepared_dirs(self) -> list[str]:
        return [self.default_output_dir]

    def _prepare_raw(self, raw_source: str) -> AdapterResult:
        prof = PROFILES[self.target_key]
        src = raw_source or self.default_raw
        bundle = pickle.load(open(src, "rb"))
        stays = bundle["stays"]

        mm_to_vital = {v: k for k, v in VITAL_TO_MM.items()}
        cond_vitals = [mm_to_vital[slot] for slot in prof.cond_vitals]
        used_vitals = [prof.channel] + [v for v in cond_vitals if v != prof.channel]

        items = list(stays.items())
        subj_of = {wid: int(s["subject_id"]) for wid, s in items}
        tr_subj, va_subj, te_subj = _subject_split(list(subj_of.values()))

        def split_of(wid) -> str:
            sj = subj_of[wid]
            return "train" if sj in tr_subj else "val" if sj in va_subj else "test"

        acc: dict[str, list[np.ndarray]] = {v: [] for v in used_vitals}
        for wid, s in items:
            if split_of(wid) != "train":
                continue
            for v in used_vitals:
                srckey, idx = VITAL_SOURCE[v]
                val, msk = _chan(s["timeseries"], srckey, idx)
                obs = msk > 0
                if obs.any():
                    acc[v].append(val[obs])
        stats = {}
        for v in used_vitals:
            allv = np.concatenate(acc[v]) if acc[v] else np.array([0.0], np.float32)
            stats[v] = (float(np.mean(allv)), float(np.std(allv) + 1e-6))

        tmean, tstd = stats[prof.channel]

        splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        test_seen: dict[int, int] = {}
        for wid, s in items:
            sp = split_of(wid)
            ts = s["timeseries"]
            tval, tmask = _chan(ts, prof.channel, prof.channel_index)
            if float(tmask.sum()) <= 0.0:
                continue
            if sp == "test":
                sj = subj_of[wid]
                if test_seen.get(sj, 0) >= self._max_test_windows_per_subject:
                    continue
                test_seen[sj] = test_seen.get(sj, 0) + 1

            target_z = _zscore(tval, tmask, tmean, tstd)

            extras: dict[str, Any] = {
                "subject_id": int(s["subject_id"]),
                "stay_id": int(s.get("stay_id", wid)),
                "careunit": str(s.get("metadata", {}).get("first_careunit", "UNK")),
                "in_hospital_mortality": int(
                    s.get("labels", {}).get("in_hospital_mortality", 0) or 0
                ),
                "icu_mortality": int(s.get("labels", {}).get("icu_mortality", 0) or 0),
                "los_days": float(s.get("labels", {}).get("los_days", 0.0) or 0.0),
                "target_signal": prof.name,
            }
            for v in cond_vitals:
                slot = VITAL_TO_MM[v]
                srckey, idx = VITAL_SOURCE[v]
                cval, cmask = _chan(ts, srckey, idx)
                cm, cs = stats[v]
                extras[f"mm_{slot}"] = _zscore(cval, cmask, cm, cs).reshape(-1, 1)
                extras[f"mm_{slot}_p"] = cmask.reshape(-1, 1).astype(np.float32)

            rec = standardize_record(
                target_z,
                tmask,
                person_id=int(s["subject_id"]),
                split=sp,
                label=0,
                dataset_name=self.name,
                extras=extras,
            )
            splits[sp].append(rec)

        for sp in ("train", "val", "test"):
            if not splits[sp]:
                raise ValueError(f"MIMIC adapter produced no {sp} records from {src}")

        supported = [prof.name] + list(prof.cond_vitals)
        metadata = DatasetMetadata(
            name=self.name,
            display_name=f"MIMIC {prof.name} ({prof.units})",
            description=f"MIMIC ICU {prof.name} imputation from {os.path.basename(src)}; "
            f"subject-disjoint splits; native {prof.units} z-score.",
            raw_source=src,
            source_dir=self.default_output_dir,
            sampling_minutes=self._sampling_minutes,
            day_len=self._day_len,
            eval_window_start=0,
            eval_window_end=self._day_len,
            glucose_mean_mgdl=tmean,
            glucose_std_mgdl=tstd,
            glucose_normalization=f"native_{prof.name}_zscore",
            aligned_to_midnight=False,
            supported_modalities=supported,
            cohort_counts={"icu_windows": sum(len(splits[s]) for s in splits)},
            notes={
                "split_summary": summarize_splits(splits),
                "native_mean": tmean,
                "native_std": tstd,
                "units": prof.units,
                "vital_stats": stats,
                "nmar_lo": prof.nmar_lo,
                "nmar_hi": prof.nmar_hi,
                "mar_covariate": prof.mar_covariate,
                "cond_vitals": list(prof.cond_vitals),
                "burden": {
                    "normal_lo": prof.normal_lo,
                    "normal_hi": prof.normal_hi,
                    "low_thr": prof.nmar_lo,
                    "high_thr": prof.nmar_hi,
                },
            },
        )
        return AdapterResult(
            train=splits["train"],
            val=splits["val"],
            test=splits["test"],
            metadata=metadata,
        )


class MimicAbpAdapter(_MimicPickleAdapter):
    name = "mimic_abp"
    aliases = ("mimic_iii_abp", "mimic3_abp")
    target_key = "abp_mean"
    default_raw = str(DATA_ROOT / "raw" / "mimic3_waveform_abp.pkl")


class MimicHrAdapter(_MimicPickleAdapter):
    name = "mimic_hr"
    aliases = ("mimic_iii_hr", "mimic3_hr")
    target_key = "hr"
    default_raw = str(DATA_ROOT / "raw" / "mimic3_waveform_abp.pkl")


class MimicIvHrAdapter(_MimicPickleAdapter):
    name = "mimic_iv_hr"
    aliases = ("mimic4_hr",)
    target_key = "hr_hourly"
    default_raw = str(DATA_ROOT / "raw" / "mimic_abp_sample.pkl")
    _day_len = 24
    _sampling_minutes = 60
