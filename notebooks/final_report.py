# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Final Performance Report
#
# This notebook assembles the latest evaluation metrics and curated figures into a publishable report.

# %% tags=["parameters"]
# Parameters
metrics_path = "results/metrics_latest.json"
figures_dir = "results/figures"

# %%
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from IPython.display import Image, Markdown, display
import matplotlib.pyplot as plt

from pipeline.evaluate import build_cumulative_returns

metrics_path = Path(metrics_path)
figures_dir = Path(figures_dir)
report_generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

if not metrics_path.exists():
    raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

figures_dir.mkdir(parents=True, exist_ok=True)


# %%
def _load_metrics(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.json_normalize(data).T.rename(columns={0: "value"})
        raise TypeError(f"Unsupported JSON structure: {type(data)}")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported metrics format: {suffix}")

metrics_df = _load_metrics(metrics_path)
metrics_df.index.name = metrics_df.index.name or "metric"
metrics_df.columns = [str(col) for col in metrics_df.columns]

display(Markdown("## Key Metrics"))
display(metrics_df)


# %%
def _iter_figures(directory: Path) -> Iterable[Path]:
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.svg"):
        yield from sorted(directory.glob(suffix))

figure_paths = list(_iter_figures(figures_dir))
if figure_paths:
    display(Markdown("## Figures"))
    for figure_path in figure_paths:
        title = figure_path.stem.replace("_", " ").title()
        display(Markdown(f"### {title}"))
        display(Image(filename=str(figure_path)))
else:
    display(Markdown("## Figures\n\n_No figures available for this run._"))


# %%
reports_root = metrics_path.parent
metrics_dir = reports_root / "metrics"
metadata_dir = reports_root / "metadata"


def _load_metrics_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.suffix == ".csv":
        return pd.read_csv(path, index_col=0)
    if path.suffix == ".json":
        try:
            return pd.read_json(path, orient="table")
        except ValueError:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return pd.json_normalize(payload).T.rename(columns={0: "value"})
    return None


def _collect_model_metrics() -> pd.DataFrame | None:
    sources = {
        "Baseline": [
            metrics_dir / "replication_metrics.csv",
            metrics_dir / "replication_metrics.json",
        ],
        "Transformer": [
            metrics_dir / "itrafo_metrics.csv",
            metrics_dir / "itrafo_metrics.json",
        ],
    }
    frames: list[pd.DataFrame] = []
    for label, paths in sources.items():
        for candidate in paths:
            table = _load_metrics_table(candidate)
            if table is not None and not table.empty:
                tidy = table.copy()
                tidy.index.name = tidy.index.name or "series"
                tidy.insert(0, "model", label)
                frames.append(tidy.reset_index())
                break
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    return combined.set_index(["model", "series"]).sort_index()


def _read_panel(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if frame.empty:
        return frame
    first_col = frame.columns[0]
    if isinstance(first_col, str) and "__" in first_col:
        splits = [col.split("__") for col in frame.columns]
        width = max(len(part) for part in splits)
        padded = [part + [None] * (width - len(part)) for part in splits]
        arrays = [list(level) for level in zip(*padded, strict=False)]
        names = ["series", *[f"level_{idx}" for idx in range(1, width)]]
        frame.columns = pd.MultiIndex.from_arrays(arrays, names=names)
    return frame


def _mean_series(frame: pd.DataFrame, series: str) -> pd.Series:
    data = frame.xs(series, axis=1, level=0)
    if isinstance(data, pd.Series):
        return data
    return data.mean(axis=1)


def _maybe_overlay_curves() -> None:
    metrics_table = _collect_model_metrics()
    baseline_meta = metadata_dir / "evaluation.json"
    itrafo_meta = metadata_dir / "itrafo_evaluation.json"
    baseline_panel = None
    itrafo_panel = None

    for meta_path, target in ((baseline_meta, "baseline"), (itrafo_meta, "itrafo")):
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        panel_path = Path(payload.get("inputs", {}).get("panel", ""))
        panel = _read_panel(panel_path)
        if panel is None or panel.empty:
            continue
        if target == "baseline":
            baseline_panel = panel
        else:
            itrafo_panel = panel

    comparison_available = metrics_table is not None

    overlay_available = baseline_panel is not None and itrafo_panel is not None

    if not comparison_available and not overlay_available:
        return

    display(Markdown("## Transformer vs Baseline"))

    if comparison_available:
        display(Markdown("### Metrics"))
        display(metrics_table)

    if overlay_available:
        baseline_cum = build_cumulative_returns(baseline_panel)
        itrafo_cum = build_cumulative_returns(itrafo_panel)
        target_curve = _mean_series(baseline_cum, "target")
        base_curve = _mean_series(baseline_cum, "replica")
        itrafo_curve = _mean_series(itrafo_cum, "replica")

        overlay = pd.concat(
            {
                "Target": target_curve,
                "Baseline": base_curve,
                "Transformer": itrafo_curve,
            },
            axis=1,
        ).dropna(how="all")

        fig, ax = plt.subplots(figsize=(10, 6))
        overlay.plot(ax=ax)
        ax.set_title("Cumulative Returns: Transformer vs Baseline")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return")
        ax.legend()
        fig.tight_layout()
        display(fig)
        plt.close(fig)

    if not overlay_available:
        display(Markdown("_Transformer comparison figure unavailable._"))


_maybe_overlay_curves()


# %%
display(Markdown(f"Report generated at **{report_generated_at}**"))
