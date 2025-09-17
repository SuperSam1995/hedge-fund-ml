"""Autoencoder model utilities extracted from research notebooks.

This module provides a small, configuration-driven autoencoder training stack
that mirrors the logic originally prototyped in ``03_autoencoder.ipynb``.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:  # pragma: no cover - optional runtime guard for GPU-free environments
    import tensorflow as _tf

    _tf.config.set_visible_devices([], "GPU")
except (ImportError, RuntimeError, ValueError):
    pass

from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.losses import MeanSquaredError
from keras.models import load_model
from keras.optimizers import Adam
from keras.regularizers import L2, Regularizer
from pydantic import BaseModel, Field
from sklearn.preprocessing import MinMaxScaler

from hedge_fund_ml import collect_run_metadata, set_global_seed

_DEFAULT_PACKAGES: list[str] = [
    "numpy",
    "pandas",
    "scikit-learn",
    "tensorflow",
    "keras",
]


def _ensure_frame(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError("Expected 2D array for autoencoder training data")
    columns = [f"feature_{idx}" for idx in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)


class AutoencoderModelConfig(BaseModel):
    """Architecture hyper-parameters for the autoencoder."""

    latent_dim: int = Field(default=8, gt=0)
    hidden_dims: list[int] = Field(default_factory=lambda: [64, 32])
    activation: str = "relu"
    latent_activation: str | None = None
    output_activation: str | None = None
    l2: float = Field(default=0.0, ge=0.0)
    input_dim: int | None = Field(default=None, gt=0)

    model_config = {"extra": "forbid", "validate_assignment": True}

    def with_input_dim(self, dimension: int) -> "AutoencoderModelConfig":
        if dimension <= 0:
            raise ValueError("input dimension must be positive")
        if self.input_dim is not None and self.input_dim != dimension:
            raise ValueError(
                "Configured input_dim does not match training data dimension"
            )
        return self.model_copy(update={"input_dim": dimension})


class AutoencoderTrainingConfig(BaseModel):
    """Training loop configuration."""

    epochs: int = Field(default=5, gt=0)
    batch_size: int = Field(default=64, gt=0)
    patience: int = Field(default=2, gt=0)
    min_delta: float = Field(default=0.0, ge=0.0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    verbose: int = Field(default=1, ge=0)

    model_config = {"extra": "forbid"}


class AutoencoderOutputConfig(BaseModel):
    """Artefact persistence settings."""

    root: Path = Field(default=Path("models/ae"))
    run_path: Path | None = None

    model_config = {"extra": "forbid"}

    def prepare_run_dir(self) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        destination = self.root / timestamp
        destination.mkdir(parents=True, exist_ok=False)
        return destination

    def resolve_run_path(self) -> Path:
        if self.run_path is not None:
            return self.run_path
        if not self.root.exists():
            raise FileNotFoundError(f"No autoencoder artefacts found in {self.root}")
        candidates = sorted(path for path in self.root.iterdir() if path.is_dir())
        if not candidates:
            raise FileNotFoundError(f"No autoencoder artefacts found in {self.root}")
        return candidates[-1]


class AutoencoderDataConfig(BaseModel):
    """Dataset configuration for running the autoencoder pipeline."""

    features_path: Path
    validation_fraction: float = Field(default=0.2, ge=0.0, lt=1.0)
    shuffle: bool = False
    dropna: bool = True

    model_config = {"extra": "forbid"}


class AutoencoderConfig(BaseModel):
    """Top-level configuration consumed by the training pipeline."""

    seed: int = 42
    packages: list[str] = Field(default_factory=lambda: list(_DEFAULT_PACKAGES))
    model: AutoencoderModelConfig = Field(default_factory=AutoencoderModelConfig)
    training: AutoencoderTrainingConfig = Field(
        default_factory=AutoencoderTrainingConfig
    )
    output: AutoencoderOutputConfig = Field(default_factory=AutoencoderOutputConfig)
    data: AutoencoderDataConfig | None = None

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: Path | str) -> "AutoencoderConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(payload)


@dataclass(slots=True)
class AutoencoderArtifacts:
    """Bundle of in-memory handles and artefact locations."""

    model: Model
    scaler: MinMaxScaler
    history: pd.DataFrame
    metrics: pd.DataFrame
    run_dir: Path


def _build_regularizer(value: float) -> Regularizer | None:
    if value == 0.0:
        return None
    return L2(value)


def build_model(cfg: AutoencoderConfig) -> Model:
    """Instantiate and compile an autoencoder according to ``cfg``."""

    model_cfg = cfg.model
    if model_cfg.input_dim is None:
        raise ValueError("model.input_dim must be specified before building")

    regularizer = _build_regularizer(model_cfg.l2)
    inputs = Input(shape=(model_cfg.input_dim,), name="features")
    x = inputs
    for index, units in enumerate(model_cfg.hidden_dims):
        x = Dense(
            units,
            activation=model_cfg.activation,
            kernel_regularizer=regularizer,
            name=f"encoder_{index}",
        )(x)
    latent = Dense(
        model_cfg.latent_dim,
        activation=model_cfg.latent_activation,
        kernel_regularizer=regularizer,
        name="latent",
    )(x)
    x = latent
    for index, units in enumerate(reversed(model_cfg.hidden_dims)):
        x = Dense(
            units,
            activation=model_cfg.activation,
            kernel_regularizer=regularizer,
            name=f"decoder_{index}",
        )(x)
    outputs = Dense(
        model_cfg.input_dim,
        activation=model_cfg.output_activation,
        kernel_regularizer=regularizer,
        name="reconstruction",
    )(x)
    autoencoder = Model(inputs=inputs, outputs=outputs, name="autoencoder")
    optimizer = Adam(learning_rate=cfg.training.learning_rate)
    autoencoder.compile(
        optimizer=optimizer,
        loss=MeanSquaredError(name="reconstruction_loss"),
        metrics=[MeanSquaredError(name="mse")],
    )
    return autoencoder


def _prepare_frames(
    X_train: pd.DataFrame | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = _ensure_frame(X_train)
    val_frame = _ensure_frame(X_val)
    missing = set(train_frame.columns) - set(val_frame.columns)
    extra = set(val_frame.columns) - set(train_frame.columns)
    if missing or extra:
        raise ValueError(
            "Training and validation data must contain the same feature columns"
        )
    val_frame = val_frame.loc[:, train_frame.columns]
    return train_frame, val_frame


def _write_history(history: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(path, index_label="epoch")


def _write_metrics(metrics: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(path, index=False)


def fit(
    X_train: pd.DataFrame | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    cfg: AutoencoderConfig,
) -> AutoencoderArtifacts:
    """Train the autoencoder and persist artefacts."""

    set_global_seed(cfg.seed)
    train_frame, val_frame = _prepare_frames(X_train, X_val)

    cfg = cfg.model_copy(
        update={
            "model": cfg.model.with_input_dim(train_frame.shape[1]),
        }
    )
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_frame.to_numpy(dtype=np.float32))
    val_scaled = scaler.transform(val_frame.to_numpy(dtype=np.float32))

    model = build_model(cfg)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.training.patience,
            min_delta=cfg.training.min_delta,
            restore_best_weights=True,
        )
    ]
    history_obj = model.fit(
        train_scaled,
        train_scaled,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        validation_data=(val_scaled, val_scaled),
        verbose=cfg.training.verbose,
        shuffle=True,
        callbacks=callbacks,
    )
    history = pd.DataFrame(history_obj.history)

    run_dir = cfg.output.prepare_run_dir()
    model.save(run_dir / "model.keras")
    with (run_dir / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "columns.json").write_text(
        json.dumps({"columns": list(train_frame.columns)}),
        encoding="utf-8",
    )
    metadata = collect_run_metadata(seed=cfg.seed, packages=cfg.packages)
    metadata.write_json(run_dir / "metadata.json")

    _write_history(history, run_dir / "history.csv")

    train_eval = model.evaluate(
        train_scaled,
        train_scaled,
        verbose=0,
        return_dict=True,
    )
    val_eval = model.evaluate(val_scaled, val_scaled, verbose=0, return_dict=True)
    metrics = pd.DataFrame(
        [
            {"split": "train", **train_eval},
            {"split": "val", **val_eval},
        ]
    )
    _write_metrics(metrics, run_dir / "metrics.csv")

    return AutoencoderArtifacts(
        model=model,
        scaler=scaler,
        history=history,
        metrics=metrics,
        run_dir=run_dir,
    )


def transform(
    X: pd.DataFrame | np.ndarray,
    cfg: AutoencoderConfig,
) -> pd.DataFrame:
    """Project ``X`` into the learned latent space using saved artefacts."""

    run_dir = cfg.output.resolve_run_path()
    columns_path = run_dir / "columns.json"
    if not columns_path.exists():
        raise FileNotFoundError(f"Missing column metadata at {columns_path}")
    columns_payload = json.loads(columns_path.read_text(encoding="utf-8"))
    expected_columns = list(columns_payload.get("columns", []))
    if not expected_columns:
        raise ValueError("Saved autoencoder columns metadata is empty")

    frame = _ensure_frame(X)
    missing = set(expected_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Input data missing columns: {sorted(missing)}")
    frame = frame.loc[:, expected_columns]

    with (run_dir / "scaler.pkl").open("rb") as handle:
        scaler: MinMaxScaler = pickle.load(handle)
    model = load_model(run_dir / "model.keras")
    encoder = Model(inputs=model.input, outputs=model.get_layer("latent").output)
    scaled = scaler.transform(frame.to_numpy(dtype=np.float32))
    latent = encoder.predict(scaled, verbose=0)
    columns = [f"latent_{idx}" for idx in range(latent.shape[1])]
    return pd.DataFrame(latent, index=frame.index, columns=columns)


__all__ = [
    "AutoencoderArtifacts",
    "AutoencoderConfig",
    "AutoencoderDataConfig",
    "AutoencoderModelConfig",
    "AutoencoderOutputConfig",
    "AutoencoderTrainingConfig",
    "build_model",
    "fit",
    "transform",
]
