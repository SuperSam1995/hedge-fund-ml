"""Data catalogue utilities.

The registry keeps the on-disk datasets immutable and discoverable.  It reads a
YAML configuration through a Pydantic schema, giving us validation and a single
source of truth for asset metadata.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterator, Literal, cast

import yaml
from pydantic import BaseModel, Field

_STAGE = Literal["raw", "interim", "processed"]


class DatasetConfig(BaseModel):
    stage: _STAGE
    filename: str
    description: str = ""
    checksum: str | None = Field(
        default=None,
        description="Optional SHA256 checksum used to freeze raw datasets.",
    )

    model_config = {
        "extra": "forbid",
    }


class RegistryConfig(BaseModel):
    datasets: Dict[str, DatasetConfig]

    model_config = {
        "extra": "forbid",
    }

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RegistryConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
            config: RegistryConfig = cls.model_validate(payload)
            return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DataRegistry:
    """Lightweight directory aware registry for project datasets."""

    def __init__(self, root: Path | str, config: RegistryConfig) -> None:
        self.root = Path(root)
        self.config = config
        for stage in ("raw", "interim", "processed"):
            (self.root / stage).mkdir(parents=True, exist_ok=True)

    def datasets(self) -> Iterator[str]:
        yield from self.config.datasets.keys()

    def path(self, name: str) -> Path:
        dataset = self.config.datasets[name]
        return self.root / dataset.stage / dataset.filename

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def checksum(self, name: str) -> str:
        target = self.path(name)
        if not target.exists():
            raise FileNotFoundError(target)
        return _sha256(target)

    def verify(self, name: str) -> bool:
        dataset = self.config.datasets[name]
        if dataset.checksum is None:
            return True
        return dataset.checksum == self.checksum(name)

    @classmethod
    def from_yaml(cls, root: Path | str, config_path: Path | str) -> "DataRegistry":
        config = RegistryConfig.from_yaml(config_path)
        return cls(root=root, config=config)


__all__ = ["DatasetConfig", "RegistryConfig", "DataRegistry"]
