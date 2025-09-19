from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module() -> object:
    module_path = Path("scripts/bundle_report.py")
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def test_manifest_entries_include_captions(tmp_path):
    module = _load_module()
    bundle = module.BundleResult(  # type: ignore[attr-defined]
        manifest=tmp_path / "manifest.json",
        context=tmp_path / "context.json",
        index=tmp_path / "index.md",
        tables=[tmp_path / "tables" / "metrics.csv"],
        figures=[tmp_path / "figures" / "metrics.png"],
        configs=[tmp_path / "configs" / "eval.yaml"],
    )

    entries = bundle.entries(tmp_path)

    kinds = {(entry.kind, entry.path) for entry in entries}
    assert ("manifest", "manifest.json") in kinds
    assert ("table", "tables/metrics.csv") in kinds
    for entry in entries:
        assert entry.caption
        assert entry.description
