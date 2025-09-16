"""Utilities for recording run metadata."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable


@dataclass
class RunMetadata:
    seed: int
    python_version: str
    packages: Dict[str, str]
    git_commit: str
    git_dirty: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def write_json(self, path: Path | str) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


def _package_versions(packages: Iterable[str]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def collect_run_metadata(seed: int, packages: Iterable[str]) -> RunMetadata:
    git_commit = _git("rev-parse", "HEAD") or "unknown"
    git_status = _git("status", "--porcelain")
    return RunMetadata(
        seed=seed,
        python_version=sys.version,
        packages=_package_versions(packages),
        git_commit=git_commit,
        git_dirty=bool(git_status),
    )


__all__ = ["RunMetadata", "collect_run_metadata"]
