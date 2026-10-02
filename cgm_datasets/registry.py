from __future__ import annotations

import os

from cgm_datasets.adapters import BaseDatasetAdapter, build_adapters


class DatasetRegistry:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self._by_name: dict[str, BaseDatasetAdapter] = {}
        for adapter in build_adapters(root_dir):
            self._by_name[adapter.name] = adapter
            for alias in adapter.aliases:
                self._by_name[alias] = adapter

    def get(self, dataset_name: str) -> BaseDatasetAdapter:
        key = dataset_name.strip().lower().replace("-", "_")
        if key in self._by_name:
            return self._by_name[key]
        raise KeyError(
            f"unknown dataset '{dataset_name}'. Available: {', '.join(sorted({a.name for a in self._by_name.values()}))}"
        )


def default_registry() -> DatasetRegistry:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return DatasetRegistry(root_dir)
