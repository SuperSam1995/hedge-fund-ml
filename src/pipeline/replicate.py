"""Replication pipeline orchestrating features, model and weights."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, ValidationError

from features import HKSpanModel
from hedge_fund_ml import collect_run_metadata

from .features import (
    FeatureArtifacts,
    FeatureRunConfig,
    build_features,
    persist_artifacts,
)

_DEFAULT_PACKAGES = ["numpy", "pandas", "scikit-learn", "statsmodels"]

__all__ = [
    "ReplicateConfig",
    "ReplicateOutputConfig",
    "ReplicationArtifacts",
    "ReplicationResult",
    "build_weights_panel",
    "run_replication",
]


class ReplicateOutputConfig(BaseModel):
    """Output locations produced by the replication pipeline."""

    weights: Path = Field(description="CSV storing time-series of portfolio weights.")
    scaler: Path = Field(description="CSV persistence of the volatility scaler.")
    metadata: Path | None = Field(
        default=None,
        description="Optional metadata snapshot (JSON).",
    )

    model_config = {"extra": "forbid"}


class ReplicateConfig(BaseModel):
    """Top-level configuration orchestrating features, model and weights."""

    features: FeatureRunConfig
    output: ReplicateOutputConfig
    packages: list[str] = Field(default_factory=lambda: list(_DEFAULT_PACKAGES))

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ReplicateConfig":
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
class ReplicationArtifacts:
    """Artefacts generated during replication."""

    features: FeatureArtifacts
    weights: pd.DataFrame


@dataclass(slots=True)
class ReplicationResult:
    """Final output describing persisted artefacts."""

    feature_frame: Path
    hk_model_path: Path
    weights_path: Path
    scaler_path: Path
    metadata_path: Path | None


def _expand_weights(model: HKSpanModel, index: pd.Index) -> pd.DataFrame:
    if model.state is None:  # pragma: no cover - defensive
        raise RuntimeError("HKSpanModel must be fitted before extracting weights")
    coefficients = model.state.coefficients
    expanded: dict[str, pd.DataFrame] = {}
    for target in coefficients.columns:
        panel = pd.DataFrame(
            np.tile(coefficients[target].to_numpy(), (len(index), 1)),
            index=index,
            columns=coefficients.index,
        )
        expanded[str(target)] = panel
    weights = pd.concat(expanded, axis=1)
    level_names = ["target", *coefficients.index.names]
    weights.columns = weights.columns.set_names(level_names)
    return weights


def build_weights_panel(artifacts: FeatureArtifacts) -> pd.DataFrame:
    """Construct a time-series panel of static HK span weights."""

    return _expand_weights(artifacts.hk_model, artifacts.scaled_features.index)


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    labels = ["__".join(map(str, column)) for column in frame.columns]
    flattened = frame.copy()
    flattened.columns = labels
    return flattened


def _persist_weights(path: Path, weights: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = _flatten_columns(weights)
    flattened.to_csv(path)


def run_replication(config: ReplicateConfig) -> ReplicationResult:
    """Execute the replication pipeline end-to-end."""

    feature_artifacts = build_features(config.features)
    persist_artifacts(config.features, feature_artifacts)

    weights = build_weights_panel(feature_artifacts)
    _persist_weights(config.output.weights, weights)

    scaler_path = config.output.scaler
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    feature_artifacts.scaler.save(scaler_path)

    metadata_path = config.output.metadata
    if metadata_path is not None:
        metadata = collect_run_metadata(config.features.seed, config.packages)
        payload = {
            "run": metadata.to_dict(),
            "outputs": {
                "features": str(config.features.data.output_features),
                "hk_model": str(config.features.data.output_model),
                "weights": str(config.output.weights),
                "scaler": str(scaler_path),
            },
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    return ReplicationResult(
        feature_frame=config.features.data.output_features,
        hk_model_path=config.features.data.output_model,
        weights_path=config.output.weights,
        scaler_path=scaler_path,
        metadata_path=metadata_path,
    )
