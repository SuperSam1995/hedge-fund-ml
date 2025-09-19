"""Create a self-contained bundle of the latest replication artefacts."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, Template

from hedge_fund_ml import collect_run_metadata
from pipeline.evaluate import EvaluationConfig, run_evaluation
from report import (
    build_metrics_summary,
    export_metrics_long,
    export_returns_long,
    export_weights_long,
    plot_cumulative,
    plot_rolling_te,
    plot_turnover,
)

DEFAULT_CONFIG_DIR = Path("configs")
DEFAULT_PACKAGES = ("numpy", "pandas", "matplotlib", "jinja2")
DEFAULT_EVAL_CONFIG = Path("configs/eval.yaml")

INDEX_TEMPLATE = """
# Research bundle — {{ run_id }}

Manifest created: {{ manifest_created_at }} UTC  \
Context generated: {{ context_generated_at }} UTC  \
Metrics summary generated: {{ metrics_generated_at }} UTC  \
Source commit: {{ git_commit }}{% if git_dirty %} (dirty){% endif %}

## Quick stats

{% if quick_stats %}
| metric | best | worst | mean | median | std |
|--------|------|-------|------|--------|-----|
{% for stat in quick_stats -%}
| {{stat.title}} | {{stat.best}} | {{stat.worst}} | {{stat.mean}} | {{stat.median}} | {{stat.std}} |
{% endfor %}
{% else %}
_No metrics available._
{% endif %}

## Strategies

{% if strategies %}
{% for strategy in strategies -%}
- {{ strategy }}
{% endfor %}
{% else %}
_No strategies recorded._
{% endif %}

## Artefacts

{% if artifacts %}
{% for artifact in artifacts -%}
- **{{ artifact.caption }}**  \
  Path: `{{ artifact.path }}`  \
  {{ artifact.description }}
{% endfor %}
{% else %}
_No artefacts recorded._
{% endif %}

"""


@dataclass
class ManifestEntry:
    kind: str
    path: str
    caption: str
    description: str


def _format_label(text: str) -> str:
    cleaned = text.replace("_", " ").replace("-", " ").strip()
    return cleaned.title() if cleaned else "Artifact"


def _summarise_entry(kind: str, path: Path) -> tuple[str, str]:
    stem = _format_label(path.stem)
    suffix = path.suffix.lstrip(".").upper()
    ext_label = f"{suffix} file" if suffix else "file"
    if kind == "manifest":
        return (
            "Bundle manifest",
            "Machine-readable inventory of all bundle artefacts.",
        )
    if kind == "context":
        return (
            "Execution context",
            "Run metadata including package versions and git state.",
        )
    if kind == "index":
        return (
            "Index page",
            "Markdown overview summarising key metrics and bundle contents.",
        )
    if kind == "summary":
        return (
            "Metrics summary",
            "Aggregated statistics saved as metrics_summary.json.",
        )
    if kind == "table":
        return (f"Table: {stem}", f"Tabular data exported as {ext_label} {path.name}.")
    if kind == "figure":
        return (f"Figure: {stem}", f"Static visualisation saved to {path.name}.")
    if kind == "config":
        return (
            f"Config: {stem}",
            "YAML configuration used to parameterise the run.",
        )
    return (stem, f"Bundle artefact stored in {path.name}.")


@dataclass
class BundleResult:
    manifest: Path
    context: Path
    index: Path
    metrics_summary: Path
    tables: Sequence[Path]
    figures: Sequence[Path]
    configs: Sequence[Path]

    def entries(self, root: Path) -> list[ManifestEntry]:
        def _entry(kind: str, path: Path) -> ManifestEntry:
            caption, description = _summarise_entry(kind, path)
            return ManifestEntry(
                kind=kind,
                path=str(path.relative_to(root)),
                caption=caption,
                description=description,
            )

        paths: list[ManifestEntry] = [
            _entry("manifest", self.manifest),
            _entry("context", self.context),
            _entry("index", self.index),
        ]
        paths.append(_entry("summary", self.metrics_summary))
        paths.extend(_entry("table", path) for path in self.tables)
        paths.extend(_entry("figure", path) for path in self.figures)
        paths.extend(_entry("config", path) for path in self.configs)
        return paths


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle reporting artefacts into a run folder")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination directory for the bundle (e.g. reports/bundle/2023-01-01T0000Z)",
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=DEFAULT_EVAL_CONFIG,
        help=f"Evaluation config used to locate artefacts (default: {DEFAULT_EVAL_CONFIG})",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help=f"Directory containing configuration YAML files (default: {DEFAULT_CONFIG_DIR})",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return "n/a"


def _format_series_stat(entry: dict[str, Any] | None) -> str:
    if not entry:
        return "n/a"
    series = entry.get("series")
    value = entry.get("value")
    if series is None:
        return "n/a"
    formatted = _format_float(value)
    return f"{series} ({formatted})" if formatted != "n/a" else str(series)


def _summarise_metrics(summary: dict[str, Any]) -> list[dict[str, str]]:
    metrics = summary.get("metrics", {})
    quick_stats: list[dict[str, str]] = []
    for name, payload in metrics.items():
        if not isinstance(payload, dict):
            continue
        title = payload.get("title") or _format_label(name)
        best = _format_series_stat(payload.get("best"))
        worst = _format_series_stat(payload.get("worst"))
        quick_stats.append(
            {
                "name": str(name),
                "title": str(title),
                "best": best,
                "worst": worst,
                "mean": _format_float(payload.get("mean")),
                "median": _format_float(payload.get("median")),
                "std": _format_float(payload.get("std")),
            }
        )
    return quick_stats


def _ensure_evaluation_outputs(config: EvaluationConfig) -> None:
    if config.output.metrics_csv.exists():
        return
    run_evaluation(config)


def _load_metrics_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.rename(frame.index.name or "series")
    return frame


def _load_panel(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Panel data not found: {path}") from exc
    if frame.empty:
        return frame
    first_column = frame.columns[0]
    if isinstance(first_column, str) and "__" in first_column:
        splits = [str(col).split("__") for col in frame.columns]
        width = max(len(parts) for parts in splits)
        padded = [parts + [None] * (width - len(parts)) for parts in splits]
        arrays = [list(level) for level in zip(*padded, strict=False)]
        names = ["series", *[f"level_{idx}" for idx in range(1, width)]]
        frame.columns = pd.MultiIndex.from_arrays(arrays, names=names)
    return frame


def _build_returns_panel(panel: pd.DataFrame) -> pd.DataFrame:
    series_frames: dict[str, pd.DataFrame] = {}
    column_map = {
        "target": "target",
        "replica": "hk_prediction",
        "residual": "hk_residual",
    }
    for label, column in column_map.items():
        if isinstance(panel.columns, pd.MultiIndex):
            try:
                subset = panel.xs(column, axis=1, level=0)
            except KeyError:
                continue
        else:
            if column not in panel.columns:
                continue
            subset = panel[[column]]
        if isinstance(subset, pd.Series):
            subset = subset.to_frame(name=column)
        numeric = subset.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        if numeric.empty:
            continue
        series_frames[label] = numeric.astype(float)

    if not series_frames:
        raise ValueError("No return series available in panel")

    returns_panel = pd.concat(series_frames, axis=1)
    default_names = [
        "series",
        *[f"level_{idx}" for idx in range(1, returns_panel.columns.nlevels)],
    ]
    column_names = [
        name if name is not None else default_names[idx]
        for idx, name in enumerate(returns_panel.columns.names)
    ]
    returns_panel.columns = returns_panel.columns.set_names(column_names)
    return returns_panel


def _load_weights_frame(path: Path) -> pd.DataFrame:
    try:
        weights = pd.read_csv(path, index_col=0, parse_dates=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Weights data not found: {path}") from exc
    if weights.empty:
        return weights
    first_column = weights.columns[0]
    if isinstance(first_column, str) and "__" in first_column:
        splits = [str(col).split("__") for col in weights.columns]
        width = max(len(parts) for parts in splits)
        padded = [parts + [None] * (width - len(parts)) for parts in splits]
        arrays = [list(level) for level in zip(*padded, strict=False)]
        names = [f"level_{idx}" for idx in range(width)]
        weights.columns = pd.MultiIndex.from_arrays(arrays, names=names)
    return weights.astype(float)


def _collapse_returns_for_plots(returns_panel: pd.DataFrame) -> pd.DataFrame:
    if returns_panel.empty:
        return returns_panel
    if isinstance(returns_panel.columns, pd.MultiIndex):
        collapsed: dict[str, pd.Series] = {}
        for name in returns_panel.columns.get_level_values(0).unique():
            subset = returns_panel.xs(name, axis=1, level=0)
            if isinstance(subset, pd.Series):
                series = subset
            else:
                series = subset.mean(axis=1)
            collapsed[str(name)] = series.rename("return")
        collapsed_frame = pd.concat(collapsed, axis=1)
        collapsed_frame.columns = collapsed_frame.columns.set_names(["series", "metric"])
        return collapsed_frame
    frame = returns_panel.copy()
    frame.columns = pd.MultiIndex.from_product(
        [frame.columns.astype(str), ["return"]],
        names=["series", "metric"],
    )
    return frame


def _strategy_labels(returns_frame: pd.DataFrame, weights: pd.DataFrame) -> list[str]:
    if returns_frame.empty:
        return []
    if isinstance(returns_frame.columns, pd.MultiIndex):
        returns_labels = {
            str(label)
            for label in returns_frame.columns.get_level_values(0)
            if str(label) != "target"
        }
    else:
        returns_labels = {str(label) for label in returns_frame.columns if str(label) != "target"}

    if weights.empty:
        return sorted(returns_labels)

    if isinstance(weights.columns, pd.MultiIndex):
        weight_labels = {str(label) for label in weights.columns.get_level_values(0)}
    else:
        weight_labels = {str(label) for label in weights.columns}

    intersect = returns_labels.intersection(weight_labels)
    if intersect:
        return sorted(intersect)
    return sorted(returns_labels)


def _copy_configs(source: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    if not source.exists():
        return copied
    for path in sorted(source.glob("*.yaml")):
        target = destination / path.name
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def _write_manifest(root: Path, run_id: str, entries: Iterable[ManifestEntry]) -> Path:
    manifest_path = root / "manifest.json"
    payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": [asdict(entry) for entry in entries],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _write_context(root: Path, payload: dict[str, Any]) -> Path:
    context_path = root / "context.json"
    context_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return context_path


def _render_index(
    template: Template,
    destination: Path,
    *,
    manifest: dict[str, Any],
    context: dict[str, Any],
    metrics_summary: dict[str, Any],
) -> Path:
    index_path = destination / "index.md"
    metadata = context.get("metadata", {}) if isinstance(context, dict) else {}
    git_commit = str(metadata.get("git_commit", "unknown"))
    git_dirty = bool(metadata.get("git_dirty", False))
    manifest_created_at = str(manifest.get("created_at", "unknown"))
    context_generated_at = str(context.get("generated_at", "unknown"))
    metrics_generated_at = str(metrics_summary.get("generated_at", "unknown"))
    run_id = (
        str(manifest.get("run_id"))
        if manifest.get("run_id") is not None
        else str(context.get("run_id", "unknown"))
    )
    artifacts = manifest.get("artifacts", []) if isinstance(manifest, dict) else []
    quick_stats = _summarise_metrics(metrics_summary)
    series_mapping = metrics_summary.get("series", {})
    if isinstance(series_mapping, dict):
        strategies = sorted(str(name) for name in series_mapping)
    else:
        strategies = []
    payload = template.render(
        run_id=run_id,
        manifest_created_at=manifest_created_at,
        context_generated_at=context_generated_at,
        metrics_generated_at=metrics_generated_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
        quick_stats=quick_stats,
        strategies=strategies,
        artifacts=artifacts,
    )
    index_path.write_text(payload, encoding="utf-8")
    return index_path


def _prepare_template() -> Template:
    env = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
    return env.from_string(INDEX_TEMPLATE)


def _build_context_payload(
    run_id: str,
    metadata: dict[str, Any],
    metrics_frame: pd.DataFrame,
    eval_config: EvaluationConfig,
    eval_config_path: Path,
    summary_path: Path,
    tables: Sequence[Path],
    figures: Sequence[Path],
    configs: Sequence[Path],
    bundle_root: Path,
) -> dict[str, Any]:
    def _rel(path: Path) -> str:
        return str(path.relative_to(bundle_root))

    metrics_payload = metrics_frame.to_dict(orient="index")
    artefacts = {
        "metrics_summary": _rel(summary_path),
        "tables": [_rel(path) for path in tables],
        "figures": [_rel(path) for path in figures],
        "configs": [_rel(path) for path in configs],
    }

    evaluation_payload = {
        "config_path": str(eval_config_path),
        "data": {
            "panel": str(eval_config.data.panel),
            "weights": str(eval_config.data.weights),
        },
        "outputs": {
            "metrics_csv": str(eval_config.output.metrics_csv),
            "metrics_json": str(eval_config.output.metrics_json),
            "metrics_summary": str(eval_config.output.metrics_summary),
            "figure": str(eval_config.output.figure),
        },
    }

    return {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "evaluation": evaluation_payload,
        "metrics": metrics_payload,
        "artefacts": artefacts,
    }


def bundle(out: Path, eval_config_path: Path, config_dir: Path) -> BundleResult:
    out.mkdir(parents=True, exist_ok=True)
    run_id = out.name

    eval_config = EvaluationConfig.from_yaml(eval_config_path)
    _ensure_evaluation_outputs(eval_config)

    metrics_frame = _load_metrics_frame(eval_config.output.metrics_csv)
    panel = _load_panel(eval_config.data.panel)
    returns_panel = _build_returns_panel(panel)
    weights_frame = _load_weights_frame(eval_config.data.weights)

    tables: list[Path] = []
    tables.append(export_metrics_long(metrics_frame, out))
    tables.append(export_returns_long(returns_panel, out))
    weight_tables = export_weights_long(weights_frame, out)
    tables.extend(weight_tables.values())

    returns_for_plots = _collapse_returns_for_plots(returns_panel)
    strategies = _strategy_labels(returns_for_plots, weights_frame)
    figures: list[Path] = []
    for strategy in strategies:
        try:
            figures.append(plot_cumulative(returns_for_plots, strategy, out))
        except (KeyError, ValueError):
            continue
        try:
            figures.append(plot_rolling_te(returns_for_plots, strategy, out_dir=out))
        except (KeyError, ValueError):
            pass
        try:
            figures.append(plot_turnover(weights_frame, strategy, out))
        except (KeyError, ValueError):
            pass

    metrics_long = (
        metrics_frame.copy()
        .rename_axis(index="series")
        .reset_index()
        .melt(id_vars="series", var_name="metric", value_name="value")
        .sort_values(["series", "metric"], ignore_index=True)
    )
    metrics_summary_payload = build_metrics_summary(metrics_long)
    metrics_summary_path = out / "metrics_summary.json"
    metrics_summary_path.write_text(
        json.dumps(metrics_summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    configs = _copy_configs(config_dir, out / "configs")

    metadata = collect_run_metadata(seed=0, packages=DEFAULT_PACKAGES)
    metadata_dict = metadata.to_dict()

    template = _prepare_template()
    manifest_path = out / "manifest.json"
    context_path = out / "context.json"

    bundle_layout = BundleResult(
        manifest=manifest_path,
        context=context_path,
        index=out / "index.md",
        metrics_summary=metrics_summary_path,
        tables=tables,
        figures=figures,
        configs=configs,
    )

    manifest = _write_manifest(out, run_id, bundle_layout.entries(out))
    manifest_payload = _load_json(manifest)

    context_payload = _build_context_payload(
        run_id,
        metadata_dict,
        metrics_frame,
        eval_config,
        eval_config_path,
        metrics_summary_path,
        tables,
        figures,
        configs,
        out,
    )

    index = _render_index(
        template,
        out,
        manifest=manifest_payload,
        context=context_payload,
        metrics_summary=metrics_summary_payload,
    )

    context = _write_context(out, context_payload)

    return BundleResult(
        manifest=manifest,
        context=context,
        index=index,
        metrics_summary=metrics_summary_path,
        tables=tables,
        figures=figures,
        configs=configs,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    bundle(
        args.out,
        args.eval_config,
        args.configs,
    )


if __name__ == "__main__":
    main()
