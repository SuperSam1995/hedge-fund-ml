"""CI smoke test ensuring dummy metrics artifact is created."""

from __future__ import annotations

import json
from pathlib import Path


def test_ci_smoke_generates_metrics(tmp_path: Path) -> None:
    """Create a deterministic dummy metrics payload in results/metrics."""
    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / "smoke.json"
    payload = {
        "status": "ok",
        "generated_at": "1970-01-01T00:00:00Z",
        "note": "CI smoke test",
    }

    staging_file = tmp_path / "payload.json"
    staging_file.write_text(json.dumps(payload))
    metrics_path.write_text(staging_file.read_text())

    assert metrics_path.is_file(), "Smoke metrics file was not created"
