"""Plotting utilities for report artefacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

__all__ = [
    "plot_cumulative",
    "plot_rolling_te",
    "plot_turnover",
]

plt.switch_backend("Agg")


@dataclass(frozen=True)
class _StrategyDescriptor:
    label: str
    slug: str


def _slugify(label: str) -> str:
    keep = [ch if ch.isalnum() else "_" for ch in label]
    slug = "".join(keep).strip("_")
    return slug or "strategy"


def _descriptor_from_mapping(mapping: Mapping[str, Any]) -> _StrategyDescriptor:
    label_source = next(
        (str(mapping[key]) for key in ("label", "series", "name", "id") if mapping.get(key)),
        None,
    )
    slug_source: str | None = None
    for key in ("slug", "figure", "path"):
        value = mapping.get(key)
        if value:
            slug_source = Path(value).stem if key in {"figure", "path"} else str(value)
            break
    if slug_source is None:
        slug_source = next(
            (str(mapping[key]) for key in ("series", "name", "id", "label") if mapping.get(key)),
            None,
        )
    label = label_source or slug_source or "strategy"
    slug = _slugify(slug_source or label)
    return _StrategyDescriptor(label=label, slug=slug)


def _normalise_strategy(strategy: Any) -> _StrategyDescriptor:
    if isinstance(strategy, _StrategyDescriptor):
        return strategy
    if isinstance(strategy, Path):
        label = strategy.stem
        return _StrategyDescriptor(label=label, slug=_slugify(label))
    if isinstance(strategy, str):
        return _StrategyDescriptor(label=strategy, slug=_slugify(strategy))
    if isinstance(strategy, Mapping):
        return _descriptor_from_mapping(strategy)
    if isinstance(strategy, Sequence) and not isinstance(strategy, (bytes, bytearray)):
        parts = [str(part) for part in strategy]
        joined = "_".join(parts)
        label = " / ".join(parts)
        return _StrategyDescriptor(label=label, slug=_slugify(joined))
    label = str(strategy)
    return _StrategyDescriptor(label=label, slug=_slugify(label))


def _ensure_figures_dir(out_dir: Path | str) -> Path:
    figures_dir = Path(out_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _select_series(frame: pd.DataFrame, key: str) -> pd.Series:
    if isinstance(frame, pd.Series):
        return frame.rename(key)
    if isinstance(frame.columns, pd.MultiIndex):
        try:
            subset = frame.xs(key, axis=1, level=0)
        except KeyError as exc:  # pragma: no cover - explicit handling tested indirectly
            raise KeyError(f"Series '{key}' not found in MultiIndex columns") from exc
        if isinstance(subset, pd.DataFrame):
            if subset.shape[1] != 1:
                raise ValueError(f"Expected a single column for '{key}', got {subset.shape[1]}")
            series = subset.iloc[:, 0]
        else:
            series = subset
    else:
        if key not in frame.columns:
            raise KeyError(f"Series '{key}' not in DataFrame columns")
        series = frame[key]
    return pd.Series(series, copy=True).sort_index()


def _figure_path(figures_dir: Path, descriptor: _StrategyDescriptor, suffix: str) -> Path:
    stem = descriptor.slug
    if not stem.endswith(f"_{suffix}"):
        stem = f"{stem}_{suffix}"
    return figures_dir / f"{stem}.png"


def plot_cumulative(
    series_df: pd.DataFrame,
    strategy: Any,
    out_dir: Path | str,
    dpi: int = 130,
) -> Path:
    """Plot cumulative returns for the selected strategy."""

    descriptor = _normalise_strategy(strategy)
    figures_dir = _ensure_figures_dir(out_dir)
    series = _select_series(series_df, descriptor.label)
    cumulative = (1.0 + series.fillna(0.0)).cumprod()
    cumulative = cumulative.rename("cumulative_return")

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(cumulative.index, cumulative.values, label=descriptor.label, linewidth=2.0)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.set_title(f"Cumulative returns – {descriptor.label}")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate()

    path = _figure_path(figures_dir, descriptor, "cumulative")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rolling_te(
    series_df: pd.DataFrame,
    strategy: Any,
    window: int = 24,
    out_dir: Path | str = Path("reports"),
    dpi: int = 130,
) -> Path:
    """Plot rolling tracking error between a strategy and the target series."""

    if window <= 1:
        raise ValueError("window must be greater than 1 for rolling tracking error")

    descriptor = _normalise_strategy(strategy)
    figures_dir = _ensure_figures_dir(out_dir)
    strategy_series = _select_series(series_df, descriptor.label)
    target_series = _select_series(series_df, "target")
    aligned_strategy, aligned_target = strategy_series.align(target_series, join="inner")
    diff = aligned_strategy - aligned_target
    rolling_te = diff.rolling(window=window, min_periods=window).std()

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(rolling_te.index, rolling_te.values, color="#d62728", linewidth=2.0)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling tracking error")
    ax.set_title(f"{window}-period tracking error – {descriptor.label}")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.autofmt_xdate()

    path = _figure_path(figures_dir, descriptor, f"rolling_te_{window}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_turnover(
    weights_df: pd.DataFrame,
    strategy: Any,
    out_dir: Path | str,
    dpi: int = 130,
) -> Path:
    """Plot portfolio turnover computed from period-on-period weight changes."""

    descriptor = _normalise_strategy(strategy)
    figures_dir = _ensure_figures_dir(out_dir)

    if isinstance(weights_df.columns, pd.MultiIndex):
        try:
            subset = weights_df.xs(descriptor.label, axis=1, level=0)
        except KeyError as exc:
            raise KeyError(f"Weights for strategy '{descriptor.label}' not found") from exc
        weights = subset.copy()
        if isinstance(weights, pd.Series):
            weights = weights.to_frame(name=descriptor.label)
    else:
        if descriptor.label not in weights_df.columns:
            raise KeyError(f"Weights for strategy '{descriptor.label}' not found")
        weights = weights_df[[descriptor.label]].copy()

    weights = weights.sort_index()
    turnover = 0.5 * weights.diff().abs().sum(axis=1)
    turnover = turnover.rename("turnover")

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(turnover.index, turnover.values, color="#1f77b4", linewidth=2.0)
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover")
    ax.set_title(f"Portfolio turnover – {descriptor.label}")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.autofmt_xdate()

    path = _figure_path(figures_dir, descriptor, "turnover")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
