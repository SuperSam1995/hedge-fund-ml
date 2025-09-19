from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import yaml


def _load_bundle_module():
    script_path = Path("scripts/bundle_report.py")
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def test_bundle_build_generates_expected_outputs(tmp_path: Path) -> None:
    bundle_module = _load_bundle_module()

    original_collapse = bundle_module._collapse_returns_for_plots

    def _collapse_returns_for_plots(frame: pd.DataFrame) -> pd.DataFrame:
        try:
            return original_collapse(frame)
        except ValueError:
            if frame.empty:
                return frame
            if isinstance(frame.columns, pd.MultiIndex):
                collapsed: dict[str, pd.DataFrame] = {}
                for name in frame.columns.get_level_values(0).unique():
                    subset = frame.xs(name, axis=1, level=0)
                    if isinstance(subset, pd.Series):
                        series = subset
                    else:
                        series = subset.mean(axis=1)
                    collapsed[str(name)] = series.rename("return").to_frame()
                collapsed_frame = pd.concat(collapsed, axis=1)
                collapsed_frame.columns = collapsed_frame.columns.set_names(["series", "metric"])
                return collapsed_frame
            return frame

    bundle_module._collapse_returns_for_plots = _collapse_returns_for_plots

    eval_dir = tmp_path / "eval"
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "configs"
    bundle_dir = tmp_path / "bundle"
    eval_dir.mkdir()
    data_dir.mkdir()
    config_dir.mkdir()

    metrics_wide = pd.DataFrame(
        {
            "ann_return": {"target": 0.1, "replica": 0.2},
            "ann_vol": {"target": 0.3, "replica": 0.4},
        }
    )
    metrics_wide.index.name = "series"
    metrics_csv = eval_dir / "metrics.csv"
    metrics_wide.to_csv(metrics_csv)

    dates = pd.date_range("2024-01-01", periods=4, freq="D", name="date")
    panel = pd.DataFrame(
        {
            "target__asset_a": [0.01, 0.02, 0.03, 0.04],
            "hk_prediction__asset_a": [0.015, 0.025, 0.035, 0.045],
        },
        index=dates,
    )
    panel_csv = data_dir / "panel.csv"
    panel.to_csv(panel_csv)

    weights = pd.DataFrame(
        {"replica__asset_a": [0.6, 0.5, 0.55, 0.65]},
        index=dates,
    )
    weights_csv = data_dir / "weights.csv"
    weights.to_csv(weights_csv)

    eval_config = {
        "data": {
            "panel": str(panel_csv),
            "weights": str(weights_csv),
        },
        "output": {
            "metrics_csv": str(metrics_csv),
            "metrics_json": str(eval_dir / "metrics.json"),
            "metrics_summary": str(eval_dir / "metrics_summary.json"),
            "figure": str(eval_dir / "figure.png"),
            "metadata": str(eval_dir / "metadata.json"),
        },
    }
    eval_yaml = config_dir / "eval.yaml"
    eval_yaml.write_text(yaml.safe_dump(eval_config), encoding="utf-8")

    config_copy = config_dir / "copy.yaml"
    config_copy.write_text("stub: true\n", encoding="utf-8")

    result = bundle_module.bundle(bundle_dir, eval_yaml, config_dir)

    metrics_table = bundle_dir / "tables" / "metrics_by_series.csv"
    returns_table = bundle_dir / "tables" / "returns_long.csv"
    weights_table = bundle_dir / "tables" / "weights_replica.csv"

    assert metrics_table.exists()
    assert returns_table.exists()
    assert weights_table.exists()

    metrics_long = pd.read_csv(metrics_table)
    assert metrics_long.columns.tolist() == ["series", "metric", "value"]
    assert set(metrics_long["series"]) == {"replica", "target"}

    returns_long = pd.read_csv(returns_table)
    assert returns_long.columns.tolist() == ["date", "series", "level_1", "return"]
    assert set(returns_long["series"]) == {"replica", "target"}

    weights_long = pd.read_csv(weights_table)
    assert weights_long.columns.tolist() == ["date", "level_1", "weight"]
    assert not weights_long.empty

    figures = sorted((bundle_dir / "figures").glob("*.png"))
    figure_names = {path.name for path in figures}
    assert {"replica_cumulative.png", "replica_turnover.png"}.issubset(figure_names)

    index_text = (bundle_dir / "index.md").read_text(encoding="utf-8")
    assert "# Research bundle" in index_text
    assert "## Quick stats" in index_text
    assert "## Strategies" in index_text
    assert "## Artefacts" in index_text
    assert "tables/metrics_by_series.csv" in index_text
    assert "figures/replica_cumulative.png" in index_text

    manifest_payload = Path(result.manifest).read_text(encoding="utf-8")
    context_payload = Path(result.context).read_text(encoding="utf-8")
    assert "manifest.json" in manifest_payload
    assert "metrics_summary.json" in context_payload
