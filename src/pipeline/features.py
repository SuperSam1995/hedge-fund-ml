"""Reusable feature engineering pipeline extracted from the CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from features import (
    HKSpanConfig,
    HKSpanModel,
    ReturnsBuilder,
    ReturnsConfig,
    ReturnsDataset,
    VolatilityScaleConfig,
    VolatilityScaler,
)
from hedge_fund_ml import collect_run_metadata, set_global_seed

__all__ = [
    "DataPaths",
    "FeatureArtifacts",
    "FeatureRunConfig",
    "SplitConfig",
    "build_features",
    "load_feature_config",
    "persist_artifacts",
]


def _read_panel(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    frame = frame.sort_index()
    return frame


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    flattened = [
        "__".join(str(part) for part in column if part not in (None, ""))
        for column in frame.columns
    ]
    result = frame.copy()
    result.columns = flattened
    return result


class DataPaths(BaseModel):
    """Locations for the raw inputs and generated artefacts."""

    factors: Path = Field(description="Input CSV containing factor / ETF returns.")
    targets: Path = Field(description="Input CSV containing hedge fund returns.")
    output_features: Path = Field(
        description="Destination CSV for engineered features and outputs."
    )
    output_model: Path = Field(
        description="Destination JSON for the HK span model coefficients."
    )
    metadata: Path | None = Field(
        default=None,
        description="Optional location for run metadata (JSON).",
    )

    model_config = {"extra": "forbid"}


class SplitConfig(BaseModel):
    """Calendar split used to fit the span without leakage."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp | None = None
    test_end: pd.Timestamp | None = None

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    @field_validator("train_start", "train_end", "test_start", "test_end", mode="before")
    @classmethod
    def _parse_timestamp(cls, value: object) -> pd.Timestamp | None:
        if value is None:
            return None
        return pd.Timestamp(value)

    @model_validator(mode="after")
    def _check_order(self) -> "SplitConfig":
        if self.train_end < self.train_start:
            raise ValueError("train_end must be on or after train_start")
        if self.test_start is not None and self.test_start <= self.train_end:
            raise ValueError("test_start must be after train_end to avoid leakage")
        if self.test_end is not None and self.test_start is not None:
            if self.test_end < self.test_start:
                raise ValueError("test_end must be on or after test_start")
        return self

    def train_mask(self, index: pd.Index) -> pd.Series:
        mask = index >= self.train_start
        mask &= index <= self.train_end
        return mask

    def test_mask(self, index: pd.Index) -> pd.Series:
        start = self.test_start
        if start is None:
            future = index[index > self.train_end]
            if future.empty:
                raise ValueError("Unable to infer test_start from index")
            start = future.min()
        mask = index >= start
        if self.test_end is not None:
            mask &= index <= self.test_end
        return mask


class FeatureRunConfig(BaseModel):
    """Full configuration for running the feature pipeline."""

    seed: int = 42
    packages: list[str] = Field(
        default_factory=lambda: ["numpy", "pandas", "scikit-learn", "statsmodels"]
    )
    data: DataPaths
    split: SplitConfig
    returns: ReturnsConfig
    volatility: VolatilityScaleConfig = Field(default_factory=VolatilityScaleConfig)
    hk_span: HKSpanConfig = Field(default_factory=HKSpanConfig)

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: Path | str) -> "FeatureRunConfig":
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


@dataclass
class FeatureArtifacts:
    """In-memory bundle returned by :func:`build_features`."""

    dataset: ReturnsDataset
    scaled_features: pd.DataFrame
    hk_model: HKSpanModel
    scaler: VolatilityScaler


def build_features(config: FeatureRunConfig) -> FeatureArtifacts:
    """Run the deterministic feature pipeline."""

    set_global_seed(config.seed)
    factors = _read_panel(config.data.factors)
    targets = _read_panel(config.data.targets)

    builder = ReturnsBuilder(config.returns)
    dataset = builder.build(features=factors, target=targets)

    index = dataset.features.index
    train_mask = config.split.train_mask(index)
    if not train_mask.any():
        raise ValueError("Training window produced no observations")
    test_mask = config.split.test_mask(index)
    if not test_mask.any():
        raise ValueError("Test window produced no observations")

    scaler = VolatilityScaler(config.volatility)
    scaler.fit(dataset.features.loc[train_mask])
    full_scaled = scaler.transform(dataset.features)

    hk_model = HKSpanModel(config.hk_span)
    hk_model.fit(full_scaled.loc[train_mask], dataset.target.loc[train_mask])

    return FeatureArtifacts(
        dataset=dataset,
        scaled_features=full_scaled,
        hk_model=hk_model,
        scaler=scaler,
    )


def persist_artifacts(config: FeatureRunConfig, artifacts: FeatureArtifacts) -> None:
    """Persist engineered features and the spanning model to disk."""

    predictions = artifacts.hk_model.predict(artifacts.scaled_features)
    residuals = artifacts.dataset.target - predictions
    frame = pd.concat(
        {
            "features": artifacts.scaled_features,
            "target": artifacts.dataset.target,
            "hk_prediction": predictions,
            "hk_residual": residuals,
        },
        axis=1,
    )
    flattened = _flatten_columns(frame)

    output_path = config.data.output_features
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flattened.to_csv(output_path, index=True)

    artifacts.hk_model.dump(config.data.output_model)

    metadata_path = config.data.metadata
    if metadata_path is not None:
        metadata = collect_run_metadata(config.seed, config.packages)
        payload = {
            "run": metadata.to_dict(),
            "rows": {
                "total": int(len(artifacts.dataset.features)),
                "train": int(config.split.train_mask(artifacts.dataset.features.index).sum()),
                "test": int(config.split.test_mask(artifacts.dataset.features.index).sum()),
            },
            "outputs": {
                "features": str(output_path),
                "hk_model": str(config.data.output_model),
            },
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


def load_feature_config(path: Path | None) -> FeatureRunConfig:
    """Load a feature configuration from YAML for CLI reuse."""

    target = path or Path("configs/features.yaml")
    return FeatureRunConfig.from_yaml(target)
