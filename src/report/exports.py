"""CSV export helpers for evaluation artefacts."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

__all__ = [
    "export_metrics_long",
    "export_returns_long",
    "export_weights_long",
]


def _ensure_tables_dir(out_dir: Path | str) -> Path:
    path = Path(out_dir) / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stack_frame(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    tidy = frame.copy()
    index_name = tidy.index.name or "date"
    tidy.index = tidy.index.rename(index_name)

    if isinstance(tidy.columns, pd.MultiIndex):
        names = [
            name if name is not None else f"level_{idx}"
            for idx, name in enumerate(tidy.columns.names)
        ]
        tidy.columns = tidy.columns.set_names(names)
        stacked = tidy.stack(names)
    else:
        column_name = tidy.columns.name or "series"
        tidy.columns = tidy.columns.rename(column_name)
        stacked = tidy.stack()

    return stacked.rename(value_name).reset_index()


def export_metrics_long(df_metrics: pd.DataFrame, out_dir: Path | str) -> Path:
    """Write metrics in long format to ``tables/metrics_by_series.csv``."""

    tables_dir = _ensure_tables_dir(out_dir)
    metrics_long = (
        df_metrics.copy()
        .rename_axis(index="series")
        .reset_index()
        .melt(id_vars="series", var_name="metric", value_name="value")
        .sort_values(["series", "metric"], ignore_index=True)
    )
    output = tables_dir / "metrics_by_series.csv"
    metrics_long.to_csv(output, index=False)
    return output


def export_returns_long(df_series: pd.DataFrame, out_dir: Path | str) -> Path:
    """Write return series in long format to ``tables/returns_long.csv``."""

    tables_dir = _ensure_tables_dir(out_dir)
    returns_long = _stack_frame(df_series, value_name="return")
    output = tables_dir / "returns_long.csv"
    returns_long.to_csv(output, index=False)
    return output


def _slugify(label: object) -> str:
    text = str(label)
    slug = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return slug or "strategy"


def export_weights_long(df_weights: pd.DataFrame, out_dir: Path | str) -> dict[str, Path]:
    """Write portfolio weights in long format under ``tables/``.

    Returns
    -------
    dict[str, Path]
        Mapping from strategy identifier to the CSV path that was written.
    """

    tables_dir = _ensure_tables_dir(out_dir)
    weights = df_weights.copy()
    weights.index = weights.index.rename(weights.index.name or "date")

    written: dict[str, Path] = {}

    if isinstance(weights.columns, pd.MultiIndex):
        names = [
            name if name is not None else f"level_{idx}"
            for idx, name in enumerate(weights.columns.names)
        ]
        weights.columns = weights.columns.set_names(names)
        strategies = weights.columns.get_level_values(0).unique()
        for strategy in strategies:
            subset = weights.xs(strategy, axis=1, level=0)
            if isinstance(subset, pd.Series):
                subset = subset.to_frame(name=strategy)
            weights_long = _stack_frame(subset, value_name="weight")
            path = tables_dir / f"weights_{_slugify(strategy)}.csv"
            weights_long.to_csv(path, index=False)
            written[str(strategy)] = path
        return written

    weights_long = _stack_frame(weights, value_name="weight")
    label = weights.columns.name or "portfolio"
    path = tables_dir / f"weights_{_slugify(label)}.csv"
    weights_long.to_csv(path, index=False)
    written[str(label)] = path
    return written
