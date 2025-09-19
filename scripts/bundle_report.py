"""Create a self-contained bundle of the latest replication artefacts."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from jinja2 import Environment, Template
from matplotlib import pyplot as plt

from hedge_fund_ml import collect_run_metadata

DEFAULT_METRICS_PATH = Path("reports/metrics_latest.json")
DEFAULT_FIGURE_DIR = Path("reports/figures")
DEFAULT_CONFIG_DIR = Path("configs")
DEFAULT_PACKAGES = ("numpy", "pandas", "matplotlib", "jinja2")

INDEX_TEMPLATE = """
# Research bundle — {{ run_id }}

Generated: {{ created_at }} UTC  \
Source commit: {{ metadata.git_commit[:7] }}{% if metadata.git_dirty %} (dirty){% endif %}

## Metrics snapshot

| metric | value |
|--------|-------|
{% for metric, value in metrics.items() -%}
| {{ metric }} | {{ "%.4g"|format(value) }} |
{% endfor %}

## Contents

- Configuration files: {{ config_count }}
- Tables: {{ table_count }}
- Figures: {{ figure_count }}

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
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Metrics JSON path (default: {DEFAULT_METRICS_PATH})",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help=f"Directory with source figures (default: {DEFAULT_FIGURE_DIR})",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help=f"Directory containing configuration YAML files (default: {DEFAULT_CONFIG_DIR})",
    )
    return parser.parse_args(argv)


def _load_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _export_tables(metrics: dict[str, float], destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    if not metrics:
        empty = destination / "metrics.csv"
        pd.DataFrame(columns=["metric", "value"]).to_csv(empty, index=False)
        return [empty]
    frame = pd.DataFrame([metrics])
    tidy = frame.melt(var_name="metric", value_name="value")
    table_path = destination / "metrics.csv"
    tidy.to_csv(table_path, index=False)
    return [table_path]


def _render_figures(metrics: dict[str, float], destination: Path, sources: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    if not metrics:
        fallback = destination / "placeholder.png"
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No metrics", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(fallback, dpi=200)
        plt.close(fig)
        created.append(fallback)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        names = list(metrics)
        values = [metrics[name] for name in names]
        ax.bar(names, values)
        ax.set_title("Key metrics")
        ax.set_ylabel("Value")
        ax.set_xlabel("Metric")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        figure_path = destination / "metrics.png"
        fig.savefig(figure_path, dpi=200)
        plt.close(fig)
        created.append(figure_path)
    if sources.exists():
        for path in sorted(sources.glob("*.png")):
            target = destination / path.name
            shutil.copy2(path, target)
            created.append(target)
    return created


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


def _write_context(
    root: Path,
    run_id: str,
    metadata: dict[str, object],
    metrics: dict[str, float],
) -> Path:
    context_path = root / "context.json"
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "metrics": metrics,
    }
    context_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return context_path


def _render_index(
    template: Template,
    destination: Path,
    *,
    run_id: str,
    created_at: str,
    metadata: dict[str, object],
    metrics: dict[str, float],
    config_count: int,
    table_count: int,
    figure_count: int,
) -> Path:
    index_path = destination / "index.md"
    payload = template.render(
        run_id=run_id,
        created_at=created_at,
        metadata=metadata,
        metrics=metrics,
        config_count=config_count,
        table_count=table_count,
        figure_count=figure_count,
    )
    index_path.write_text(payload, encoding="utf-8")
    return index_path


def _prepare_template() -> Template:
    env = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
    return env.from_string(INDEX_TEMPLATE)


def bundle(out: Path, metrics_path: Path, figure_dir: Path, config_dir: Path) -> BundleResult:
    out.mkdir(parents=True, exist_ok=True)
    run_id = out.name
    metrics = _load_metrics(metrics_path)
    tables = _export_tables(metrics, out / "tables")
    figures = _render_figures(metrics, out / "figures", figure_dir)
    configs = _copy_configs(config_dir, out / "configs")
    metadata = collect_run_metadata(seed=0, packages=DEFAULT_PACKAGES)
    metadata_dict = metadata.to_dict()
    created_at = datetime.now(timezone.utc).isoformat()
    context = _write_context(out, run_id, metadata_dict, metrics)
    template = _prepare_template()
    index = _render_index(
        template,
        out,
        run_id=run_id,
        created_at=created_at,
        metadata=metadata_dict,
        metrics=metrics,
        config_count=len(configs),
        table_count=len(tables),
        figure_count=len(figures),
    )
    bundle_paths = BundleResult(
        manifest=out / "manifest.json",
        context=context,
        index=index,
        tables=tables,
        figures=figures,
        configs=configs,
    )
    manifest = _write_manifest(out, run_id, bundle_paths.entries(out))
    return BundleResult(
        manifest=manifest,
        context=context,
        index=index,
        tables=tables,
        figures=figures,
        configs=configs,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    bundle(args.out, args.metrics, args.figures, args.configs)


if __name__ == "__main__":
    main()
