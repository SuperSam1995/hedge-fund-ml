from __future__ import annotations

from hedge_fund_ml.telemetry.metadata import collect_run_metadata


def test_collect_run_metadata_roundtrip(tmp_path) -> None:
    metadata = collect_run_metadata(seed=11, packages=["numpy"])
    assert metadata.seed == 11
    assert "numpy" in metadata.packages
    assert metadata.git_commit
    path = tmp_path / "metadata.json"
    metadata.write_json(path)
    assert path.exists()
