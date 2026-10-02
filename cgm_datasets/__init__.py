from __future__ import annotations

from cgm_datasets.common import DatasetMetadata
from cgm_datasets.registry import default_registry


def load_dataset_splits(dataset_name: str = "aireadi", prepared_dir: str | None = None):
    adapter = default_registry().get(dataset_name)
    result = adapter.load(prepared_dir=prepared_dir)
    return result.train, result.val, result.test, result.metadata


def prepare_dataset(
    dataset_name: str,
    raw_source: str,
    output_dir: str | None = None,
    *,
    force: bool = False,
) -> str:
    adapter = default_registry().get(dataset_name)
    return adapter.prepare(raw_source=raw_source, output_dir=output_dir, force=force)


def list_datasets() -> list[str]:
    registry = default_registry()
    return sorted({adapter.name for adapter in registry._by_name.values()})


__all__ = [
    "DatasetMetadata",
    "load_dataset_splits",
    "prepare_dataset",
    "list_datasets",
]
