from __future__ import annotations

from pathlib import Path

import pytest

from hedge_fund_ml.data.registry import DataRegistry


@pytest.fixture(scope="module")
def registry() -> DataRegistry:
    return DataRegistry.from_yaml("data", "configs/data_registry.yaml")


def test_registry_lists_datasets(registry: DataRegistry) -> None:
    datasets = set(registry.datasets())
    assert "etf_data" in datasets
    assert registry.exists("etf_data")


def test_registry_verifies_checksums(registry: DataRegistry) -> None:
    assert registry.verify("etf_data")


def test_registry_missing_dataset(tmp_path: Path) -> None:
    registry = DataRegistry.from_yaml(tmp_path, "configs/data_registry.yaml")
    with pytest.raises(FileNotFoundError):
        registry.checksum("etf_data")
