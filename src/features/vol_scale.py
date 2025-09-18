"""Volatility scaling utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame, Series
from pydantic import BaseModel, Field

__all__ = ["VolatilityScaleConfig", "VolatilityScaler"]


class VolatilityScaleConfig(BaseModel):
    """Configuration describing how to estimate feature volatility."""

    ddof: int = Field(1, ge=0, description="Degrees of freedom for variance estimate.")
    min_std: float = Field(
        1e-8,
        ge=0.0,
        description="Floor applied to the estimated standard deviation.",
    )

    model_config = {"extra": "forbid"}


class VolatilityScaler:
    """Global standard-deviation based feature scaling."""

    def __init__(self, config: VolatilityScaleConfig) -> None:
        self.config = config
        self.scale_: Series | None = None

    def fit(self, data: DataFrame) -> VolatilityScaler:
        if data.empty:
            raise ValueError("data must contain at least one observation")
        std = data.std(ddof=self.config.ddof)
        std = std.clip(lower=self.config.min_std)
        self.scale_ = std
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if self.scale_ is None:
            raise RuntimeError("VolatilityScaler must be fitted before transform")
        missing = set(self.scale_.index) - set(data.columns)
        if missing:
            raise KeyError(f"Missing columns for scaling: {sorted(missing)}")
        columns = list(self.scale_.index)
        return data.loc[:, columns].div(self.scale_, axis=1)

    def fit_transform(self, data: DataFrame) -> DataFrame:
        return self.fit(data).transform(data)

    def save(self, path: Path) -> None:
        if self.scale_ is None:
            raise RuntimeError("Cannot persist scaler before fitting")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.scale_.to_frame(name="scale")
        payload.to_csv(target)

    @classmethod
    def load(
        cls, path: Path | str, config: VolatilityScaleConfig | None = None
    ) -> VolatilityScaler:
        frame = pd.read_csv(path, index_col=0)
        if frame.shape[1] != 1:
            raise ValueError("Serialized scaler must contain exactly one column")
        series = frame.iloc[:, 0]
        series.name = "scale"
        active_config = config if config is not None else VolatilityScaleConfig.model_validate({})
        scaler = cls(active_config)
        scaler.scale_ = series
        return scaler
