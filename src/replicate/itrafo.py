"""Cost-aware replication driver for iTraFo forecasts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, Field, ValidationError, field_validator

from hedge_fund_ml import collect_run_metadata, set_global_seed

from .decoder import WeightDecoder

__all__ = [
    "DecoderHyperParams",
    "ITrafoColumns",
    "ITrafoConfig",
    "ITrafoPaths",
    "ITrafoRunResult",
    "run_itrafo_replication",
]


class ITrafoPaths(BaseModel):
    """File-system layout for iTraFo replication."""

    itrafo_forecast_csv: Path
    etf_forecast_csv: Path
    weights_csv: Path
    series_csv: Path
    metadata_json: Path | None = Field(default=None)

    model_config = {"extra": "forbid"}


class ITrafoColumns(BaseModel):
    """Column names required by the replication decoder."""

    date: str
    strategy: str
    yhat: str
    etfs: list[str]

    model_config = {"extra": "forbid"}

    @field_validator("etfs")
    @classmethod
    def _ensure_non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("etfs must contain at least one ticker")
        return value


class DecoderHyperParams(BaseModel):
    """Hyper-parameters steering the quadratic decoder."""

    leverage: float
    lambda_to: float = Field(default=0.0)
    lambda_l2: float = Field(default=0.0)
    long_only: bool = Field(default=False)
    solver: str = Field(default="OSQP")
    solver_opts: Mapping[str, float | int] | None = Field(default=None)

    model_config = {"extra": "forbid"}

    @field_validator("leverage")
    @classmethod
    def _positive_leverage(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("leverage must be positive")
        return value

    @field_validator("lambda_to", "lambda_l2")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("penalties must be non-negative")
        return value


class ITrafoConfig(BaseModel):
    """Validated configuration for running the iTraFo decoder."""

    seed: int = Field(default=0, ge=0)
    packages: list[str] = Field(default_factory=lambda: ["numpy", "pandas", "cvxpy"])
    paths: ITrafoPaths
    cols: ITrafoColumns
    hyper: DecoderHyperParams

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: Path | str) -> ITrafoConfig:
        try:
            payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:  # pragma: no cover - CLI error path
            raise SystemExit(f"Config file not found: {path}") from exc
        except yaml.YAMLError as exc:  # pragma: no cover - CLI error path
            raise SystemExit(f"Invalid YAML in {path}: {exc}") from exc
        try:
            return cls.model_validate(payload or {})
        except ValidationError as exc:  # pragma: no cover - CLI error path
            raise SystemExit(str(exc)) from exc

    def build_decoder(self) -> WeightDecoder:
        return WeightDecoder(
            leverage=self.hyper.leverage,
            lambda_to=self.hyper.lambda_to,
            lambda_l2=self.hyper.lambda_l2,
            long_only=self.hyper.long_only,
            solver=self.hyper.solver,
            solver_opts=self.hyper.solver_opts,
        )


@dataclass(slots=True)
class ITrafoRunResult:
    """Outputs of the iTraFo replication driver."""

    weights: pd.DataFrame
    series: pd.DataFrame
    metadata_path: Path | None


def _prepare_frame(config: ITrafoConfig) -> pd.DataFrame:
    paths = config.paths
    cols = config.cols

    forecast = pd.read_csv(paths.itrafo_forecast_csv)
    etf_forecast = pd.read_csv(paths.etf_forecast_csv)

    for frame in (forecast, etf_forecast):
        if cols.date not in frame:
            raise KeyError(f"Missing date column '{cols.date}' in inputs")
        frame[cols.date] = pd.to_datetime(frame[cols.date], utc=True)

    merged = forecast.merge(
        etf_forecast[[cols.date, *cols.etfs]],
        on=cols.date,
        how="inner",
        validate="many_to_one",
    )

    if cols.strategy not in merged or cols.yhat not in merged:
        raise KeyError("Strategy or yhat column missing from forecasts")

    missing = [ticker for ticker in cols.etfs if ticker not in merged]
    if missing:
        raise KeyError(f"Missing ETF columns: {missing}")

    merged.sort_values([cols.strategy, cols.date], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def _decode_panel(config: ITrafoConfig, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = config.cols
    decoder = config.build_decoder()

    weights_rows: list[dict[str, object]] = []
    series_rows: list[dict[str, object]] = []

    for strategy, group in frame.groupby(cols.strategy, sort=False):
        previous_weights: NDArray[np.float_] | None = None

        etf_forecasts = group[cols.etfs].to_numpy(dtype=float)
        yhats = group[cols.yhat].to_numpy(dtype=float)
        dates = group[cols.date].to_numpy()

        for i in range(len(group)):
            etf_forecast = etf_forecasts[i]
            yhat = float(yhats[i])
            date = dates[i]

            result = decoder.solve_once(etf_forecast, yhat, previous_weights)
            weights_vector: NDArray[np.float_] = result.weights

            weight_record: dict[str, object] = {cols.date: date, cols.strategy: strategy}
            weight_record.update(dict(zip(cols.etfs, weights_vector, strict=True)))
            weights_rows.append(weight_record)

            portfolio_hat = float(np.dot(etf_forecast, weights_vector))
            series_rows.append(
                {
                    cols.date: date,
                    cols.strategy: strategy,
                    "portfolio_return_hat": portfolio_hat,
                    "target_return_hat": yhat,
                }
            )
            previous_weights = weights_vector

    weights = pd.DataFrame(weights_rows)
    series = pd.DataFrame(series_rows)
    return weights, series


def _persist_outputs(config: ITrafoConfig, result: ITrafoRunResult) -> None:
    result.weights.sort_values([config.cols.strategy, config.cols.date], inplace=True)
    result.series.sort_values([config.cols.strategy, config.cols.date], inplace=True)

    config.paths.weights_csv.parent.mkdir(parents=True, exist_ok=True)
    config.paths.series_csv.parent.mkdir(parents=True, exist_ok=True)
    result.weights.to_csv(config.paths.weights_csv, index=False)
    result.series.to_csv(config.paths.series_csv, index=False)

    if config.paths.metadata_json is not None:
        metadata = collect_run_metadata(config.seed, config.packages)
        payload = {
            "run": metadata.to_dict(),
            "outputs": {
                "weights": str(config.paths.weights_csv),
                "series": str(config.paths.series_csv),
            },
        }
        config.paths.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        config.paths.metadata_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


def run_itrafo_replication(config: ITrafoConfig) -> ITrafoRunResult:
    """Execute the monthly decoding pipeline end-to-end."""

    set_global_seed(config.seed)
    frame = _prepare_frame(config)
    weights, series = _decode_panel(config, frame)
    result = ITrafoRunResult(
        weights=weights,
        series=series,
        metadata_path=config.paths.metadata_json,
    )
    _persist_outputs(config, result)
    return result
