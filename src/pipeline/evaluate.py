"""Evaluation pipeline turning replication artefacts into metrics and plots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, cast

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from pydantic import BaseModel, Field, ValidationError

from eval import (
    annualised_return,
    annualised_volatility,
    certainty_equivalent,
    max_drawdown,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    turnover,
)
from hedge_fund_ml import collect_run_metadata
from report import metrics_table

__all__ = [
    "EvaluationConfig",
    "EvaluationOutputConfig",
    "EvaluationResult",
    "build_cumulative_returns",
    "run_evaluation",
]


class EvaluationOutputConfig(BaseModel):
    """Locations for evaluation artefacts."""

    metrics_csv: Path
    metrics_json: Path
    figure: Path
    metadata: Path | None = None

    model_config = {"extra": "forbid"}


class EvaluationDataConfig(BaseModel):
    """Input artefacts required for evaluation."""

    panel: Path = Field(description="CSV exported by the replication pipeline.")
    weights: Path = Field(description="CSV containing portfolio weights over time.")

    model_config = {"extra": "forbid"}


class EvaluationSettings(BaseModel):
    periods_per_year: int = 12
    risk_free_rate: float = 0.0
    risk_aversion: float = 3.0
    omega_threshold: float = 0.0

    model_config = {"extra": "forbid"}


class EvaluationConfig(BaseModel):
    """Top level evaluation configuration."""

    data: EvaluationDataConfig
    output: EvaluationOutputConfig
    settings: EvaluationSettings = Field(default_factory=EvaluationSettings)
    packages: list[str] = Field(
        default_factory=lambda: [
            "numpy",
            "pandas",
            "matplotlib",
        ]
    )

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: Path | str) -> "EvaluationConfig":
        try:
            payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:  # pragma: no cover - CLI handles error
            raise SystemExit(f"Config file not found: {path}") from exc
        except yaml.YAMLError as exc:  # pragma: no cover - CLI handles error
            raise SystemExit(f"Invalid YAML in {path}: {exc}") from exc
        try:
            return cls.model_validate(payload or {})
        except ValidationError as exc:  # pragma: no cover - CLI handles error
            raise SystemExit(str(exc)) from exc


@dataclass(slots=True)
class EvaluationResult:
    metrics: pd.DataFrame
    cumulative_returns: pd.DataFrame


def _read_panel(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if "__" in frame.columns[0]:
        splits = [col.split("__") for col in frame.columns]
        width = max(len(part) for part in splits)
        padded = [part + [None] * (width - len(part)) for part in splits]
        arrays = [list(level) for level in zip(*padded)]
        names = ["series", *[f"level_{idx}" for idx in range(1, width)]]
        frame.columns = pd.MultiIndex.from_arrays(arrays, names=names)
    return frame


def _select_series(frame: pd.DataFrame, series: str) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        if series not in frame.columns.get_level_values(0):
            raise KeyError(f"Series '{series}' not found in panel")
        result = frame.xs(series, axis=1, level=0)
        if isinstance(result, pd.Series):
            name = result.name if result.name is not None else series
            return result.to_frame(name=name)
        return result
    raise ValueError("Expected MultiIndex columns with a top-level series label")


def _to_float_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = cast(pd.DataFrame, frame.apply(pd.to_numeric, errors="raise"))
    return numeric.astype(float)


def build_cumulative_returns(panel: pd.DataFrame) -> pd.DataFrame:
    target = _to_float_frame(_select_series(panel, "target"))
    replica = _to_float_frame(_select_series(panel, "hk_prediction"))
    cumulative = pd.concat(
        {
            "target": (1 + target).cumprod() - 1,
            "replica": (1 + replica).cumprod() - 1,
        },
        axis=1,
    )
    level_names = ["series", *target.columns.names]
    cumulative.columns = cumulative.columns.set_names(level_names)
    return cumulative


def _compute_metrics(
    panel: pd.DataFrame,
    weights: pd.DataFrame,
    settings: EvaluationSettings,
) -> Dict[str, Dict[str, float]]:
    target = _to_float_frame(_select_series(panel, "target"))
    replica = _to_float_frame(_select_series(panel, "hk_prediction"))
    residual = _to_float_frame(_select_series(panel, "hk_residual"))

    metrics: Dict[str, Dict[str, float]] = {}
    for name, data in {
        "target": target,
        "replica": replica,
        "residual": residual,
    }.items():
        for column in data.columns:
            series = data[column]
            key = f"{name}:{column}"
            metrics[key] = {
                "ann_return": annualised_return(series, settings.periods_per_year),
                "ann_vol": annualised_volatility(series, settings.periods_per_year),
                "sharpe": sharpe_ratio(
                    series,
                    risk_free_rate=settings.risk_free_rate,
                    periods_per_year=settings.periods_per_year,
                ),
                "sortino": sortino_ratio(
                    series,
                    risk_free_rate=settings.risk_free_rate,
                    periods_per_year=settings.periods_per_year,
                ),
                "max_dd": max_drawdown(series),
                "ceq": certainty_equivalent(
                    series,
                    risk_aversion=settings.risk_aversion,
                    periods_per_year=settings.periods_per_year,
                ),
                "omega": omega_ratio(series, threshold=settings.omega_threshold),
            }
    metrics["turnover"] = {"value": turnover(weights)}
    return metrics


def _persist_metrics(config: EvaluationConfig, metrics_frame: pd.DataFrame) -> None:
    config.output.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(config.output.metrics_csv)
    config.output.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    config.output.metrics_json.write_text(
        metrics_frame.to_json(orient="table", indent=2),
        encoding="utf-8",
    )


def _persist_figure(config: EvaluationConfig, cumulative: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for series in cumulative.columns.get_level_values(0).unique():
        data = cumulative.xs(series, axis=1, level=0)
        if isinstance(data, pd.Series):
            data = data.to_frame(name=series)
        mean_series = data.mean(axis=1)
        mean_series.plot(ax=ax, label=series.capitalize())
    ax.set_title("Cumulative returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.legend()
    fig.tight_layout()
    config.output.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.output.figure)
    plt.close(fig)


def run_evaluation(config: EvaluationConfig) -> EvaluationResult:
    panel = _read_panel(config.data.panel)
    weights = pd.read_csv(config.data.weights, index_col=0, parse_dates=True)
    if weights.shape[1] > 0 and "__" in weights.columns[0]:
        splits = [col.split("__") for col in weights.columns]
        width = max(len(part) for part in splits)
        padded = [part + [None] * (width - len(part)) for part in splits]
        arrays = [list(level) for level in zip(*padded)]
        names = [f"level_{idx}" for idx in range(width)]
        weights.columns = pd.MultiIndex.from_arrays(arrays, names=names)
    metrics_dict = _compute_metrics(panel, weights.astype(float), config.settings)
    metrics_frame = metrics_table(metrics_dict)
    _persist_metrics(config, metrics_frame)

    cumulative = build_cumulative_returns(panel)
    _persist_figure(config, cumulative)

    if config.output.metadata is not None:
        metadata = collect_run_metadata(0, config.packages)
        payload = {
            "run": metadata.to_dict(),
            "inputs": {
                "panel": str(config.data.panel),
                "weights": str(config.data.weights),
            },
            "outputs": {
                "metrics_csv": str(config.output.metrics_csv),
                "metrics_json": str(config.output.metrics_json),
                "figure": str(config.output.figure),
            },
        }
        config.output.metadata.parent.mkdir(parents=True, exist_ok=True)
        config.output.metadata.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    return EvaluationResult(metrics=metrics_frame, cumulative_returns=cumulative)
