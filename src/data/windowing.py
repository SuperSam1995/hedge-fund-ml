"""Utilities to build rolling windows from panel datasets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DatetimeIndex
from pydantic import BaseModel, Field, model_validator

__all__ = [
    "PanelConfig",
    "SplitConfig",
    "WindowingConfig",
    "WindowedDataset",
    "ZScoreScaler",
    "load_panels",
    "make_windows",
]


class PanelConfig(BaseModel):
    """Configuration describing the feature and target panels."""

    panel_csv: Path
    target_csv: Path
    date_col: str = Field(default="Date")
    freq: Literal["M"] = Field(default="M")
    lookback: int
    horizon: int
    standardize: Literal["zscore", "none"] = Field(default="none")
    targets: Sequence[str]

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_window_params(self) -> PanelConfig:
        if self.lookback <= 0:
            msg = "lookback must be a positive integer"
            raise ValueError(msg)
        if self.horizon <= 0:
            msg = "horizon must be a positive integer"
            raise ValueError(msg)
        if not self.targets:
            msg = "targets must contain at least one column name"
            raise ValueError(msg)
        return self


class SplitConfig(BaseModel):
    """Calendar-based train/validation/test split definition."""

    train_end: date
    val_end: date
    test_end: date

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_monotonic(self) -> SplitConfig:
        train_end_ts = pd.Timestamp(self.train_end)
        val_end_ts = pd.Timestamp(self.val_end)
        test_end_ts = pd.Timestamp(self.test_end)
        if not (train_end_ts < val_end_ts < test_end_ts):
            msg = "Require train_end < val_end < test_end"
            raise ValueError(msg)
        return self

    @property
    def train_end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.train_end)

    @property
    def val_end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.val_end)

    @property
    def test_end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.test_end)


class WindowingConfig(BaseModel):
    """Top level schema combining panel and split configuration."""

    data: PanelConfig
    split: SplitConfig

    model_config = {"extra": "forbid"}


@dataclass(frozen=True, slots=True)
class WindowedDataset:
    """Windowed representation of panel data."""

    features: np.ndarray
    targets: np.ndarray
    feature_index: DatetimeIndex
    target_index: DatetimeIndex

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            msg = "features must have shape (n_windows, lookback, n_features)"
            raise ValueError(msg)
        if self.targets.ndim != 2:
            msg = "targets must have shape (n_windows, n_targets)"
            raise ValueError(msg)
        if self.features.shape[0] != self.targets.shape[0]:
            msg = "features and targets must share the same number of windows"
            raise ValueError(msg)
        if len(self.feature_index) != self.features.shape[0]:
            msg = "feature_index length must match number of windows"
            raise ValueError(msg)
        if len(self.target_index) != self.targets.shape[0]:
            msg = "target_index length must match number of windows"
            raise ValueError(msg)


@dataclass(slots=True)
class ZScoreScaler:
    """Simple z-score scaler working on 3D windowed arrays."""

    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, data: np.ndarray) -> ZScoreScaler:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim < 2:
            msg = "data must have at least 2 dimensions"
            raise ValueError(msg)
        n_features = arr.shape[-1]
        flattened = arr.reshape(-1, n_features)
        mean = flattened.mean(axis=0)
        scale = flattened.std(axis=0, ddof=0)
        scale[scale == 0.0] = 1.0
        return cls(mean_=mean, scale_=scale)

    def transform(self, data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float64)
        if arr.shape[-1] != self.mean_.shape[0]:
            msg = "data has incompatible feature dimension"
            raise ValueError(msg)
        reshape = (1,) * (arr.ndim - 1) + (-1,)
        mean = self.mean_.reshape(reshape)
        scale = self.scale_.reshape(reshape)
        return (arr - mean) / scale

    @classmethod
    def fit_transform(cls, data: np.ndarray) -> tuple[np.ndarray, ZScoreScaler]:
        scaler = cls.fit(data)
        return scaler.transform(data), scaler


def _read_panel(path: Path, date_col: str, freq: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        msg = f"{date_col!r} not found in {path}"
        raise KeyError(msg)
    df[date_col] = pd.to_datetime(df[date_col], utc=False)
    df = df.set_index(date_col).sort_index()
    if freq != "M":
        msg = f"Unsupported frequency: {freq!r}"
        raise ValueError(msg)
    month_end = df.index.to_period("M").to_timestamp("M")
    df.index = month_end
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_panels(cfg: PanelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load factor (X) and target (y) panels aligned on month end."""

    panel = _read_panel(cfg.panel_csv, cfg.date_col, cfg.freq)
    target = _read_panel(cfg.target_csv, cfg.date_col, cfg.freq)
    missing = [name for name in cfg.targets if name not in target.columns]
    if missing:
        msg = f"Targets not found: {missing}"
        raise KeyError(msg)

    panel = panel.sort_index()
    target = target.sort_index()
    target = target.loc[:, list(cfg.targets)]

    combined_index = panel.index.intersection(target.index)
    if combined_index.empty:
        msg = "No overlapping dates between panel and target datasets"
        raise ValueError(msg)

    panel_aligned = panel.loc[combined_index]
    target_aligned = target.loc[combined_index]

    stacked = pd.concat([panel_aligned, target_aligned], axis=1).dropna()
    panel_clean = stacked.loc[:, panel_aligned.columns]
    target_clean = stacked.loc[:, target_aligned.columns]
    return panel_clean, target_clean


def make_windows(X: pd.DataFrame, y: pd.DataFrame, lookback: int, horizon: int) -> WindowedDataset:
    """Convert aligned panels into rolling windows with forward targets."""

    if lookback <= 0:
        msg = "lookback must be positive"
        raise ValueError(msg)
    if horizon <= 0:
        msg = "horizon must be positive"
        raise ValueError(msg)
    if not X.index.equals(y.index):
        msg = "X and y must share the same DatetimeIndex"
        raise ValueError(msg)

    n_obs = len(X.index)
    if n_obs < lookback + horizon:
        msg = "Not enough observations to build the requested windows"
        raise ValueError(msg)

    X_values = X.to_numpy(dtype=np.float64)
    y_values = y.to_numpy(dtype=np.float64)
    timestamps = X.index

    n_windows = n_obs - lookback - horizon + 1
    n_features = X.shape[1]
    n_targets = y.shape[1]

    features = np.empty((n_windows, lookback, n_features), dtype=np.float64)
    targets = np.empty((n_windows, n_targets), dtype=np.float64)
    feature_index = []
    target_index = []

    for start in range(n_windows):
        end = start + lookback
        target_pos = end + horizon - 1
        features[start] = X_values[start:end]
        targets[start] = y_values[target_pos]
        feature_index.append(timestamps[end - 1])
        target_index.append(timestamps[target_pos])

    return WindowedDataset(
        features=features,
        targets=targets,
        feature_index=pd.DatetimeIndex(feature_index),
        target_index=pd.DatetimeIndex(target_index),
    )
