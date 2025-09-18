"""WGAN utilities extracted from the research notebook prototype.

The implementation mirrors the original ``04_wgan.ipynb`` logic but wraps it in
configuration-driven, reproducible helpers that integrate with the rest of the
package.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:  # pragma: no cover - optional runtime guard for GPU-free environments
    import tensorflow as _tf
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover - env dependent
    raise ImportError("TensorFlow (tf.keras) is required for the WGAN module") from exc

try:  # pragma: no cover - GPU-less CI
    _tf.config.set_visible_devices([], "GPU")
except (RuntimeError, ValueError):  # pragma: no cover - best effort only
    pass

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


def _ensure_array(data: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        array = data.to_numpy(dtype=np.float32)
    else:
        array = np.asarray(data, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError("Training data must be at least 2D (samples, features...)")
    return array


class WGANModelConfig(BaseModel):
    """Architecture hyper-parameters for the generator and critic."""

    latent_dim: int = Field(default=32, gt=0)
    generator_units: list[int] = Field(default_factory=lambda: [128, 128])
    critic_units: list[int] = Field(default_factory=lambda: [128, 128])
    activation: str = "relu"
    output_activation: str = "tanh"
    critic_slope: float = Field(default=0.2, gt=0)
    data_dim: int | None = Field(default=None, gt=0)

    model_config = {"extra": "forbid", "validate_assignment": True}

    def with_data_dim(self, dimension: int) -> WGANModelConfig:
        if dimension <= 0:
            raise ValueError("data dimension must be positive")
        if self.data_dim is not None and self.data_dim != dimension:
            raise ValueError(
                "Configured data_dim does not match training data dimension",
            )
        return self.model_copy(update={"data_dim": dimension})


class WGANTrainingConfig(BaseModel):
    """Training loop parameters."""

    epochs: int = Field(default=5, gt=0)
    batch_size: int = Field(default=64, gt=0)
    n_critic: int = Field(default=3, gt=0)
    clip_value: float = Field(default=0.01, gt=0)
    learning_rate: float = Field(default=5e-5, gt=0)
    patience: int = Field(default=3, gt=0)
    min_delta: float = Field(default=1e-4, ge=0.0)

    model_config = {"extra": "forbid"}


class WGANOutputConfig(BaseModel):
    """Artefact persistence settings."""

    root: Path = Field(default=Path("models/wgan"))
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
            raise FileNotFoundError(f"No WGAN artefacts found in {self.root}")
        candidates = sorted(path for path in self.root.iterdir() if path.is_dir())
        if not candidates:
            raise FileNotFoundError(f"No WGAN artefacts found in {self.root}")
        return candidates[-1]


class WGANConfig(BaseModel):
    """Top-level configuration for the WGAN pipeline."""

    seed: int = 42
    packages: list[str] = Field(default_factory=lambda: list(_DEFAULT_PACKAGES))
    model: WGANModelConfig = Field(default_factory=WGANModelConfig)
    training: WGANTrainingConfig = Field(default_factory=WGANTrainingConfig)
    output: WGANOutputConfig = Field(default_factory=WGANOutputConfig)

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: Path | str) -> WGANConfig:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(payload)


@dataclass(slots=True)
class WGANModels:
    generator: keras.Model
    critic: keras.Model
    combined: keras.Model


@dataclass(slots=True)
class WGANArtifacts:
    generator: keras.Model
    critic: keras.Model
    history: pd.DataFrame
    run_dir: Path
    scaler: MinMaxScaler
    feature_shape: tuple[int, ...]


def _wasserstein_loss(y_true, y_pred):
    return keras.backend.mean(y_true * y_pred)


def _build_generator(cfg: WGANConfig) -> keras.Model:
    model_cfg = cfg.model
    if model_cfg.data_dim is None:
        raise ValueError("model.data_dim must be specified before building")
    inputs = keras.Input(shape=(model_cfg.latent_dim,), name="latent")
    x = inputs
    for index, units in enumerate(model_cfg.generator_units):
        dense = keras.layers.Dense(
            units,
            activation=model_cfg.activation,
            name=f"gen_dense_{index}",
        )
        x = dense(x)
        norm = keras.layers.LayerNormalization(name=f"gen_norm_{index}")
        x = norm(x)
    outputs = keras.layers.Dense(
        model_cfg.data_dim,
        activation=model_cfg.output_activation,
        name="gen_output",
    )(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="generator")


def _build_critic(cfg: WGANConfig) -> keras.Model:
    model_cfg = cfg.model
    if model_cfg.data_dim is None:
        raise ValueError("model.data_dim must be specified before building")
    inputs = keras.Input(shape=(model_cfg.data_dim,), name="samples")
    x = inputs
    for index, units in enumerate(model_cfg.critic_units):
        dense = keras.layers.Dense(units, name=f"critic_dense_{index}")
        x = dense(x)
        activation = keras.layers.LeakyReLU(
            alpha=model_cfg.critic_slope,
            name=f"critic_lrelu_{index}",
        )
        x = activation(x)
    outputs = keras.layers.Dense(1, name="critic_output")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="critic")


def build_gan(cfg: WGANConfig) -> WGANModels:
    """Instantiate the generator, critic and combined models."""

    generator = _build_generator(cfg)
    critic = _build_critic(cfg)

    critic_optimizer = keras.optimizers.RMSprop(learning_rate=cfg.training.learning_rate)
    critic.compile(loss=_wasserstein_loss, optimizer=critic_optimizer)

    critic.trainable = False
    z = keras.Input(shape=(cfg.model.latent_dim,), name="gan_latent")
    generated = generator(z)
    validity = critic(generated)
    combined = keras.Model(z, validity, name="wgan")
    generator_optimizer = keras.optimizers.RMSprop(learning_rate=cfg.training.learning_rate)
    combined.compile(loss=_wasserstein_loss, optimizer=generator_optimizer)
    critic.trainable = True

    return WGANModels(generator=generator, critic=critic, combined=combined)


def _write_history(history: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(path, index_label="epoch")


def _clip_critic_weights(model: keras.Model, value: float) -> None:
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        clipped = [np.clip(weight, -value, value) for weight in weights]
        layer.set_weights(clipped)


def _prepare_training_data(
    data: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    array = _ensure_array(data)
    feature_shape = tuple(array.shape[1:])
    flat = array.reshape(array.shape[0], -1)
    return flat, feature_shape


def train_gan(data: pd.DataFrame | np.ndarray, cfg: WGANConfig) -> WGANArtifacts:
    """Train a Wasserstein GAN on ``data`` and persist artefacts."""

    set_global_seed(cfg.seed)
    flat, feature_shape = _prepare_training_data(data)

    model_cfg = cfg.model.with_data_dim(flat.shape[1])
    cfg = cfg.model_copy(update={"model": model_cfg})

    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    scaled = scaler.fit_transform(flat)

    models = build_gan(cfg)

    batch_size = min(cfg.training.batch_size, scaled.shape[0])
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    rng = np.random.default_rng(cfg.seed)
    valid = -np.ones((batch_size, 1), dtype=np.float32)
    fake = np.ones((batch_size, 1), dtype=np.float32)

    history_rows: list[dict[str, float]] = []
    best_loss = np.inf
    epochs_without_improvement = 0

    for epoch in range(cfg.training.epochs):
        critic_losses: list[float] = []
        generator_losses: list[float] = []
        batches_per_epoch = max(scaled.shape[0] // batch_size, 1)
        for _ in range(batches_per_epoch):
            for _ in range(cfg.training.n_critic):
                indices = rng.integers(0, scaled.shape[0], size=batch_size)
                real_samples = scaled[indices]
                noise = rng.normal(size=(batch_size, cfg.model.latent_dim)).astype(np.float32)

                critic_loss_real = models.critic.train_on_batch(real_samples, valid)
                generated_samples = models.generator.predict(noise, verbose=0)
                critic_loss_fake = models.critic.train_on_batch(generated_samples, fake)
                critic_loss = 0.5 * (float(critic_loss_real) + float(critic_loss_fake))
                critic_losses.append(critic_loss)
                _clip_critic_weights(models.critic, cfg.training.clip_value)

            noise = rng.normal(size=(batch_size, cfg.model.latent_dim)).astype(np.float32)
            generator_loss = models.combined.train_on_batch(noise, valid)
            generator_losses.append(float(generator_loss))

        epoch_critic_loss = float(np.mean(critic_losses)) if critic_losses else np.nan
        epoch_generator_loss = float(np.mean(generator_losses)) if generator_losses else np.nan
        history_rows.append(
            {
                "epoch": epoch,
                "critic_loss": epoch_critic_loss,
                "generator_loss": epoch_generator_loss,
            }
        )

        if epoch_critic_loss + cfg.training.min_delta < best_loss:
            best_loss = epoch_critic_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.training.patience:
                break

    history = pd.DataFrame(history_rows).set_index("epoch")

    run_dir = cfg.output.prepare_run_dir()

    models.generator.save(run_dir / "generator.keras")
    models.critic.save(run_dir / "critic.keras")

    with (run_dir / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "feature_shape.json").write_text(
        json.dumps({"feature_shape": feature_shape}),
        encoding="utf-8",
    )
    _write_history(history, run_dir / "history.csv")

    metadata = collect_run_metadata(seed=cfg.seed, packages=cfg.packages)
    metadata.write_json(run_dir / "metadata.json")

    return WGANArtifacts(
        generator=models.generator,
        critic=models.critic,
        history=history,
        run_dir=run_dir,
        scaler=scaler,
        feature_shape=feature_shape,
    )


def _load_feature_shape(path: Path) -> tuple[int, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    shape = payload.get("feature_shape", [])
    if not isinstance(shape, list):
        raise ValueError("Invalid feature_shape metadata")
    return tuple(int(dim) for dim in shape)


def _resolve_generator(
    cfg: WGANConfig,
) -> tuple[keras.Model, MinMaxScaler, tuple[int, ...]]:
    run_dir = cfg.output.resolve_run_path()
    generator = keras.models.load_model(run_dir / "generator.keras", compile=False)
    with (run_dir / "scaler.pkl").open("rb") as handle:
        scaler: MinMaxScaler = pickle.load(handle)
    feature_shape = _load_feature_shape(run_dir / "feature_shape.json")
    return generator, scaler, feature_shape


def sample(n: int, cfg: WGANConfig) -> np.ndarray:
    """Sample ``n`` synthetic sequences from a trained generator."""

    if n <= 0:
        raise ValueError("n must be positive")
    set_global_seed(cfg.seed)
    generator, scaler, feature_shape = _resolve_generator(cfg)
    noise = np.random.normal(size=(n, cfg.model.latent_dim)).astype(np.float32)
    synthetic = generator.predict(noise, verbose=0)
    restored = cast(np.ndarray, scaler.inverse_transform(synthetic))
    if feature_shape:
        reshaped = restored.reshape((n, *feature_shape))
        return reshaped
    return restored


__all__ = [
    "WGANArtifacts",
    "WGANConfig",
    "WGANModelConfig",
    "WGANOutputConfig",
    "WGANTrainingConfig",
    "build_gan",
    "sample",
    "train_gan",
]
