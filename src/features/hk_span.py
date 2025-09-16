"""Linear spanning model inspired by the Hsu-Ku spanning test."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pandas import DataFrame, Series
from pydantic import BaseModel, Field

__all__ = ["HKSpanConfig", "HKSpanModel", "HKSpanState"]


class HKSpanConfig(BaseModel):
    """Configuration for the spanning regression."""

    add_intercept: bool = Field(
        default=True, description="Whether to include an intercept in the span."
    )
    ridge_alpha: float = Field(
        default=0.0,
        ge=0.0,
        description="Optional ridge penalty applied to the normal equations.",
    )

    model_config = {"extra": "forbid"}


def _serialise_labels(labels: Iterable[Any]) -> list[Any]:
    return [list(label) if isinstance(label, tuple) else label for label in labels]


def _deserialise_labels(labels: Iterable[Any]) -> list[Any]:
    return [tuple(label) if isinstance(label, list) else label for label in labels]


@dataclass(slots=True)
class HKSpanState:
    coefficients: DataFrame
    intercept: Series

    def to_dict(self) -> dict[str, Any]:
        return {
            "coefficients": {
                "values": self.coefficients.values.tolist(),
                "index": _serialise_labels(self.coefficients.index),
                "columns": _serialise_labels(self.coefficients.columns),
            },
            "intercept": {
                "values": self.intercept.values.tolist(),
                "index": _serialise_labels(self.intercept.index),
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HKSpanState":
        coef_payload = payload["coefficients"]
        coef_index = _deserialise_labels(coef_payload["index"])
        coef_columns = _deserialise_labels(coef_payload["columns"])
        coefficients = DataFrame(
            coef_payload["values"], index=coef_index, columns=coef_columns
        )
        intercept_payload = payload["intercept"]
        intercept_index = _deserialise_labels(intercept_payload["index"])
        intercept = Series(intercept_payload["values"], index=intercept_index)
        return cls(coefficients=coefficients, intercept=intercept)


class HKSpanModel:
    """Estimate a multi-output linear span without leaking test information."""

    def __init__(self, config: HKSpanConfig) -> None:
        self.config = config
        self.state: HKSpanState | None = None

    def fit(self, span: DataFrame, target: DataFrame) -> "HKSpanModel":
        if span.empty or target.empty:
            raise ValueError("span and target must contain observations")
        aligned_span, aligned_target = span.align(target, join="inner", axis=0)
        if aligned_span.empty:
            raise ValueError("No overlapping index between span and target")

        x = aligned_span.to_numpy()
        y = aligned_target.to_numpy()
        if self.config.add_intercept:
            ones = np.ones((x.shape[0], 1))
            design = np.concatenate([ones, x], axis=1)
        else:
            design = x

        if self.config.ridge_alpha > 0:
            xtx = design.T @ design
            xtx += self.config.ridge_alpha * np.eye(xtx.shape[0])
            beta = np.linalg.solve(xtx, design.T @ y)
        else:
            beta = np.linalg.lstsq(design, y, rcond=None)[0]

        if self.config.add_intercept:
            intercept = beta[0]
            coeffs = beta[1:]
        else:
            intercept = np.zeros(beta.shape[1])
            coeffs = beta

        coef_frame = DataFrame(
            coeffs,
            index=aligned_span.columns,
            columns=aligned_target.columns,
        )
        intercept_series = Series(intercept, index=aligned_target.columns)
        self.state = HKSpanState(coefficients=coef_frame, intercept=intercept_series)
        return self

    def predict(self, span: DataFrame) -> DataFrame:
        if self.state is None:
            raise RuntimeError("HKSpanModel must be fitted before prediction")
        columns = list(self.state.coefficients.index)
        missing = set(columns) - set(span.columns)
        if missing:
            raise KeyError(f"Missing span columns: {sorted(missing)}")
        aligned = span.loc[:, columns]
        predictions = aligned.to_numpy() @ self.state.coefficients.to_numpy()
        predictions = predictions + self.state.intercept.to_numpy()
        return DataFrame(
            predictions,
            index=aligned.index,
            columns=self.state.coefficients.columns,
        )

    def residuals(self, target: DataFrame, span: DataFrame) -> DataFrame:
        predictions = self.predict(span)
        aligned_target, aligned_pred = target.align(predictions, join="inner", axis=0)
        return aligned_target - aligned_pred

    def dump(self, path: Path | str) -> None:
        if self.state is None:
            raise RuntimeError("Cannot persist unfitted model")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.model_dump(),
            "state": self.state.to_dict(),
        }
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path | str) -> "HKSpanModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config = HKSpanConfig.model_validate(payload["config"])
        model = cls(config=config)
        model.state = HKSpanState.from_dict(payload["state"])
        return model
