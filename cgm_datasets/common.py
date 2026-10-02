from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

AI_READI_GLUCOSE_MEAN_MGDL = 132.05
AI_READI_GLUCOSE_STD_MGDL = 42.33
DEFAULT_SAMPLING_MINUTES = 5
DEFAULT_DAY_LEN = 288
DEFAULT_EVAL_WINDOW = (288, 576)


@dataclass
class DatasetMetadata:
    name: str
    display_name: str
    description: str = ""
    source_dir: str | None = None
    raw_source: str | None = None
    sampling_minutes: int = DEFAULT_SAMPLING_MINUTES
    day_len: int = DEFAULT_DAY_LEN
    eval_window_start: int = DEFAULT_EVAL_WINDOW[0]
    eval_window_end: int = DEFAULT_EVAL_WINDOW[1]
    glucose_mean_mgdl: float = AI_READI_GLUCOSE_MEAN_MGDL
    glucose_std_mgdl: float = AI_READI_GLUCOSE_STD_MGDL
    glucose_normalization: str = "fixed_ai_readi_compatible"
    aligned_to_midnight: bool = True
    supported_modalities: list[str] = field(default_factory=list)
    cohort_counts: dict[str, int] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def eval_window(self) -> tuple[int, int]:
        return int(self.eval_window_start), int(self.eval_window_end)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetMetadata":
        return cls(**data)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def standardize_record(
    glucose_z: np.ndarray,
    observed_mask: np.ndarray,
    *,
    person_id: str | int,
    split: str,
    label: int = 0,
    dataset_name: str,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    glucose_z = np.asarray(glucose_z, dtype=np.float32).reshape(-1, 1)
    observed_mask = np.asarray(observed_mask, dtype=np.float32).reshape(-1, 1)
    length = int(glucose_z.shape[0])
    record = {
        "person_id": person_id,
        "label": int(label),
        "recommended_split": split,
        "irg_ts": glucose_z,
        "irg_ts_mask": observed_mask,
        "ts_tt": np.arange(length, dtype=np.float32),
        "seq_len": length,
        "dataset_name": dataset_name,
        "sampling_minutes": DEFAULT_SAMPLING_MINUTES,
    }
    if extras:
        for key, value in extras.items():
            record[key] = value
    return record


def normalize_glucose_mgdl(
    glucose_mgdl: np.ndarray,
    *,
    mean_mgdl: float = AI_READI_GLUCOSE_MEAN_MGDL,
    std_mgdl: float = AI_READI_GLUCOSE_STD_MGDL,
) -> np.ndarray:
    arr = np.asarray(glucose_mgdl, dtype=np.float32)
    return ((arr - float(mean_mgdl)) / float(std_mgdl)).astype(np.float32)


def maybe_convert_mmol_to_mgdl(glucose_values: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(glucose_values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr, False
    median = float(np.nanmedian(finite))
    if 0.0 < median < 35.0:
        return (arr * 18.0).astype(np.float32), True
    return arr, False


def train_val_test_split(
    records: list[dict[str, Any]],
    *,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, list[dict[str, Any]]]:
    participants: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        participants.setdefault(str(record["person_id"]), []).append(record)

    groups: dict[int, list[str]] = {}
    for pid, items in participants.items():
        label = int(items[0].get("label", 0))
        groups.setdefault(label, []).append(pid)

    rng = np.random.default_rng(seed)
    split_members = {"train": [], "val": [], "test": []}
    for _, pids in sorted(groups.items()):
        pids = list(pids)
        rng.shuffle(pids)
        n = len(pids)
        if n == 1:
            split_members["train"].extend(pids)
            continue
        if n == 2:
            split_members["train"].append(pids[0])
            split_members["test"].append(pids[1])
            continue
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_val = 1
            n_train = max(1, n - 2)
        split_members["train"].extend(pids[:n_train])
        split_members["val"].extend(pids[n_train : n_train + n_val])
        split_members["test"].extend(pids[n_train + n_val :])

    total_participants = sum(len(pids) for pids in split_members.values())
    if total_participants >= 3 and not split_members["val"]:
        donor = "train" if len(split_members["train"]) > 1 else "test"
        if split_members[donor]:
            split_members["val"].append(split_members[donor].pop())
    if total_participants >= 2 and not split_members["test"]:
        donor = "train" if len(split_members["train"]) > 1 else "val"
        if split_members[donor]:
            split_members["test"].append(split_members[donor].pop())

    split_lookup = {pid: split for split, pids in split_members.items() for pid in pids}
    split_records = {"train": [], "val": [], "test": []}
    for record in records:
        split = split_lookup[str(record["person_id"])]
        record["recommended_split"] = split
        split_records[split].append(record)
    return split_records


def _candidate_split_names(dataset_name: str, split: str) -> list[str]:
    base = dataset_name.replace("-", "_")
    names = [
        f"{base}_{split}.pkl",
        f"{base.lower()}_{split}.pkl",
        f"{base.upper()}_{split}.pkl",
        f"{split}.pkl",
    ]
    if base == "aireadi":
        names.extend(
            [
                f"aireadi_cgm_{split}.pkl",
                f"aireadi_cgm_full_{split}.pkl",
                f"ai_readi_{split}.pkl",
            ]
        )
    return names


def find_split_file(dataset_dir: str, dataset_name: str, split: str) -> str:
    for name in _candidate_split_names(dataset_name, split):
        path = os.path.join(dataset_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"could not find split '{split}' under {dataset_dir}")


def prepared_dir_has_splits(dataset_dir: str, dataset_name: str) -> bool:
    if not dataset_dir or not os.path.isdir(dataset_dir):
        return False
    try:
        for split in ("train", "val", "test"):
            find_split_file(dataset_dir, dataset_name, split)
    except FileNotFoundError:
        return False
    return True


def save_prepared_dataset(
    output_dir: str,
    dataset_name: str,
    splits: dict[str, list[dict[str, Any]]],
    metadata: DatasetMetadata,
) -> str:
    ensure_dir(output_dir)
    for split in ("train", "val", "test"):
        with open(os.path.join(output_dir, f"{split}.pkl"), "wb") as handle:
            pickle.dump(splits.get(split, []), handle)
    with open(
        os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(
            metadata.to_dict() | {"name": dataset_name},
            handle,
            indent=2,
            sort_keys=True,
        )
    return output_dir


def _ensure_legacy_fields(
    records: list[dict[str, Any]], dataset_name: str, split: str
) -> list[dict[str, Any]]:
    for record in records:
        record.setdefault("dataset_name", dataset_name)
        record.setdefault("recommended_split", split)
        if "seq_len" not in record and "irg_ts" in record:
            record["seq_len"] = int(np.asarray(record["irg_ts"]).shape[0])
        if "ts_tt" not in record and "seq_len" in record:
            record["ts_tt"] = np.arange(int(record["seq_len"]), dtype=np.float32)
        record.setdefault("sampling_minutes", DEFAULT_SAMPLING_MINUTES)
    return records


def load_prepared_dataset(
    dataset_dir: str, dataset_name: str
) -> tuple[list, list, list, DatasetMetadata]:
    with open(find_split_file(dataset_dir, dataset_name, "train"), "rb") as handle:
        train = pickle.load(handle)
    with open(find_split_file(dataset_dir, dataset_name, "val"), "rb") as handle:
        val = pickle.load(handle)
    with open(find_split_file(dataset_dir, dataset_name, "test"), "rb") as handle:
        test = pickle.load(handle)

    metadata_path = os.path.join(dataset_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = DatasetMetadata.from_dict(json.load(handle))
    else:
        metadata = DatasetMetadata(
            name=dataset_name, display_name=dataset_name, source_dir=dataset_dir
        )
    train = _ensure_legacy_fields(train, dataset_name, "train")
    val = _ensure_legacy_fields(val, dataset_name, "val")
    test = _ensure_legacy_fields(test, dataset_name, "test")
    return train, val, test, metadata


def summarize_splits(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split, items in splits.items():
        lengths = [int(item["seq_len"]) for item in items]
        summary[split] = {
            "n_records": len(items),
            "n_participants": len({str(item["person_id"]) for item in items}),
            "median_seq_len": int(np.median(lengths)) if lengths else 0,
        }
    return summary


def flatten_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records]
