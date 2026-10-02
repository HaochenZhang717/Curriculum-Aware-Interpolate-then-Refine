"""Pool standardized records from several datasets and balance sampling across them."""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from cgm_datasets import load_dataset_splits


def tag_domains(
    records_by_name: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Concatenate records, stamping each with an integer domain_id.

    'aireadi' (any alias starting with 'aireadi'/'ai_readi') is forced to id 0 so that
    inference/eval, which default the domain to 0, always select the AI-READI domain.
    """
    names = list(records_by_name.keys())
    names.sort(
        key=lambda n: (
            0 if n.replace("-", "_").startswith(("aireadi", "ai_readi")) else 1,
            n,
        )
    )
    pooled: list[dict[str, Any]] = []
    for dom_id, name in enumerate(names):
        for r in records_by_name[name]:
            r = dict(r)
            r["domain_id"] = dom_id
            r["dataset_name"] = name
            pooled.append(r)
    return pooled, names


def load_pooled_records(
    dataset_names: list[str],
    *,
    split: str = "train",
    data_dir: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load `split` for each dataset and return (tagged_records, domain_names)."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
    by_name: dict[str, list[dict[str, Any]]] = {}
    for name in dataset_names:
        tr, va, te, _ = load_dataset_splits(name, prepared_dir=data_dir)
        by_name[name] = {"train": tr, "val": va, "test": te}[split]
    return tag_domains(by_name)


def make_balanced_sampler(
    window_domains: np.ndarray, *, mix: str = "equal"
) -> WeightedRandomSampler:
    """Per-window sampling weights so each domain gets the desired share.

    mix='equal'        -> every domain contributes equal total mass (rare sets up-weighted)
    mix='sqrt'         -> mass proportional to sqrt(count) (gentle balancing)
    mix='proportional' -> uniform per-window (== natural dataset sizes)
    """
    window_domains = np.asarray(window_domains)
    if window_domains.size == 0:
        raise ValueError("window_domains is empty — the dataset produced no windows")
    counts = np.bincount(window_domains)
    counts = np.where(counts == 0, 1, counts)
    if mix == "proportional":
        per_domain = np.ones_like(counts, dtype=np.float64)
    elif mix == "sqrt":
        per_domain = 1.0 / np.sqrt(counts)
    elif mix == "equal":
        per_domain = 1.0 / counts
    else:
        raise ValueError(f"unknown mix '{mix}' (use equal|sqrt|proportional)")
    weights = per_domain[window_domains].astype(np.float64)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
