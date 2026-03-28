"""Return-based feature construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, Field, PositiveInt, field_validator

__all__ = ["ReturnsConfig", "ReturnsDataset", "ReturnsBuilder"]


class ReturnsConfig(BaseModel):
    """Configuration for transforming price or return panels."""

    horizon: PositiveInt = Field(
        default=1,
        description="Number of periods to shift the prediction target forward.",
    )
    lag_periods: Sequence[PositiveInt] = Field(
        default_factory=lambda: (1, 3, 6, 12),
        description="Lag periods (in observations) used as features.",
    )
    source: str = Field(
        default="returns",
        description="Input type: either 'levels' (price levels) or 'returns'.",
    )
    dropna: bool = Field(
        default=True,
        description="Drop rows with missing values after feature construction.",
    )

    model_config = {"extra": "forbid"}

    @field_validator("lag_periods")
    @classmethod
    def _validate_lags(cls, value: Sequence[int]) -> Sequence[int]:
        if not value:
            raise ValueError("lag_periods must contain at least one element")
        unique = sorted(set(int(lag) for lag in value))
        if unique[0] < 1:
            raise ValueError("lag_periods must be positive integers")
        return tuple(unique)

    @field_validator("source")
    @classmethod
    def _validate_source(cls, value: str) -> str:
        allowed = {"levels", "returns"}
        if value not in allowed:
            raise ValueError(f"source must be one of {sorted(allowed)}")
        return value


@dataclass(slots=True)
class ReturnsDataset:
    """Container storing aligned feature and target matrices."""

    features: DataFrame
    target: DataFrame

    def to_frame(self) -> DataFrame:
        """Return a single DataFrame with hierarchical columns."""

        return pd.concat({"features": self.features, "target": self.target}, axis=1)


class ReturnsBuilder:
    """Construct lagged feature matrices and forward targets from panels."""

    def __init__(self, config: ReturnsConfig) -> None:
        self.config = config

    def _compute_returns(self, data: DataFrame) -> DataFrame:
        ordered = data.sort_index()
        if self.config.source == "levels":
            returns = cast(DataFrame, np.log(ordered / ordered.shift(1)))
        else:
            returns = ordered.copy()
        return returns

    def _lagged_features(self, returns: DataFrame) -> DataFrame:
        frames: list[DataFrame] = []
        for lag in self.config.lag_periods:
            lagged = returns.shift(lag)
            lagged.columns = pd.MultiIndex.from_product(
                [returns.columns, [f"lag_{lag}"]], names=["asset", "lag"]
            )
            frames.append(lagged)
        features = pd.concat(frames, axis=1)
        features = features.sort_index(axis=1)
        features.columns = features.columns.set_names(["asset", "lag"])
        return features

    def _forward_target(self, returns: DataFrame) -> DataFrame:
        shifted = returns.shift(-int(self.config.horizon))
        shifted.columns = pd.MultiIndex.from_product(
            [returns.columns, [f"fwd_{self.config.horizon}"]],
            names=["asset", "horizon"],
        )
        shifted.columns = shifted.columns.set_names(["asset", "horizon"])
        return shifted

    def build(self, features: DataFrame, target: DataFrame | None = None) -> ReturnsDataset:
        """Return lagged features and forward-shifted targets."""

        feature_returns = self._compute_returns(features)
        lagged = self._lagged_features(feature_returns)

        if target is None:
            target_returns = feature_returns
        else:
            target_returns = self._compute_returns(target)

        forward = self._forward_target(target_returns)

        # ⚡ Bolt Optimization: Avoid expensive concat and str.startswith column searches
        # Align indexes directly and use bitwise operations for dropping NaNs
        idx = lagged.index.intersection(forward.index)
        feature_cols = lagged.loc[idx]
        target_cols = forward.loc[idx]

        if self.config.dropna:
            valid_mask = feature_cols.notna().all(axis=1) & target_cols.notna().all(axis=1)
            feature_cols = feature_cols.loc[valid_mask]
            target_cols = target_cols.loc[valid_mask]

        return ReturnsDataset(features=feature_cols, target=target_cols)
