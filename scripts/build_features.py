"""Config-driven feature construction CLI."""

from __future__ import annotations

import argparse
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

DEFAULT_CONFIG_PATH = Path("configs/features.yaml")


def _read_panel(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()
    return df


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
    factors: Path = Field(description="Input CSV containing factor / ETF returns.")
    targets: Path = Field(description="Input CSV containing hedge fund returns.")
    output_features: Path = Field(
        description="Destination CSV for engineered features."
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
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp | None = None
    test_end: pd.Timestamp | None = None

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    @field_validator(
        "train_start", "train_end", "test_start", "test_end", mode="before"
    )
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
    seed: int = 42
    packages: list[str] = Field(
        default_factory=lambda: ["numpy", "pandas", "scikit-learn"]
    )
    data: DataPaths
    split: SplitConfig
    returns: ReturnsConfig
    volatility: VolatilityScaleConfig = Field(default_factory=VolatilityScaleConfig)
    hk_span: HKSpanConfig = Field(default_factory=HKSpanConfig)

    model_config = {"extra": "forbid"}


@dataclass
class FeatureArtifacts:
    dataset: ReturnsDataset
    scaled_features: pd.DataFrame
    hk_model: HKSpanModel
    scaler: VolatilityScaler


def build_features(config: FeatureRunConfig) -> FeatureArtifacts:
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
                "train": int(
                    config.split.train_mask(artifacts.dataset.features.index).sum()
                ),
                "test": int(
                    config.split.test_mask(artifacts.dataset.features.index).sum()
                ),
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


def load_config(path: Path | None) -> FeatureRunConfig:
    target = path or DEFAULT_CONFIG_PATH
    try:
        payload = yaml.safe_load(Path(target).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Config file not found: {target}") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"Invalid YAML in {target}: {exc}") from exc
    try:
        return FeatureRunConfig.model_validate(payload or {})
    except ValidationError as exc:
        raise SystemExit(str(exc)) from exc


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build engineered features from ETF and hedge fund data"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the feature configuration YAML file.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config)
    artifacts = build_features(config)
    persist_artifacts(config, artifacts)
    print(f"Features saved to {config.data.output_features}")
    print(f"HK span model saved to {config.data.output_model}")


if __name__ == "__main__":
    main()
