"""Train the iTransformer on windowed factor data."""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pydantic import BaseModel, Field, ValidationError
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.windowing import (
    PanelConfig,
    SplitConfig,
    WindowedDataset,
    ZScoreScaler,
    load_panels,
    make_windows,
)
from hedge_fund_ml import collect_run_metadata, set_global_seed
from hedge_fund_ml.models import ITransformer, ITransformerConfig

DEFAULT_CONFIG = Path("configs/transformer.yaml")


class TransformerModelConfig(BaseModel):
    """Hyper-parameters for the iTransformer backbone."""

    d_model: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    depth: int = Field(gt=0)
    dropout: float = Field(ge=0.0, le=1.0)
    attn_bias: bool = Field(default=False)

    model_config = {"extra": "forbid"}

    def to_itransformer(self, input_dim: int, seq_len: int, target_dim: int) -> ITransformerConfig:
        if self.attn_bias:
            msg = "attn_bias is not supported by the current iTransformer implementation"
            raise ValueError(msg)
        return ITransformerConfig(
            input_dim=input_dim,
            seq_len=seq_len,
            target_dim=target_dim,
            embed_dim=self.d_model,
            depth=self.depth,
            num_heads=self.n_heads,
            dropout=self.dropout,
        )


class TrainingConfig(BaseModel):
    """Optimization controls."""

    batch_size: int = Field(gt=0)
    max_epochs: int = Field(gt=0)
    early_stopping_patience: int = Field(gt=0)
    lr: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    seed: int = Field(ge=0)

    model_config = {"extra": "forbid"}


class TransformerRunConfig(BaseModel):
    """Full configuration for the transformer training run."""

    data: PanelConfig
    split: SplitConfig
    model: TransformerModelConfig
    train: TrainingConfig
    output_root: Path = Field(default=Path("models/itrafo"))
    forecast_csv: Path = Field(default=Path("data/interim/itrafo_forecast.csv"))
    packages: list[str] = Field(
        default_factory=lambda: [
            "numpy",
            "pandas",
            "torch",
            "pytorch-lightning",
            "pyyaml",
        ]
    )

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: Path) -> TransformerRunConfig:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:  # pragma: no cover - CLI surface
            raise SystemExit(f"Config file not found: {path}") from exc
        except yaml.YAMLError as exc:  # pragma: no cover - CLI surface
            raise SystemExit(f"Invalid YAML in {path}: {exc}") from exc
        try:
            return cls.model_validate(payload or {})
        except ValidationError as exc:  # pragma: no cover - CLI surface
            raise SystemExit(str(exc)) from exc


@dataclass(slots=True)
class ArraySplits:
    """Tensor-ready arrays and indices."""

    train_features: np.ndarray
    train_targets: np.ndarray
    val_features: np.ndarray
    val_targets: np.ndarray
    test_features: np.ndarray
    test_targets: np.ndarray
    full_features: np.ndarray
    full_targets: np.ndarray
    target_index: pd.DatetimeIndex


def _subset_dataset(dataset: WindowedDataset, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = dataset.features[mask]
    targets = dataset.targets[mask]
    return features, targets


def _split_windows(dataset: WindowedDataset, split: SplitConfig) -> ArraySplits:
    target_index = dataset.target_index
    train_mask = target_index <= split.train_end_ts
    val_mask = (target_index > split.train_end_ts) & (target_index <= split.val_end_ts)
    test_mask = (target_index > split.val_end_ts) & (target_index <= split.test_end_ts)

    if not train_mask.any():
        raise ValueError("Training split is empty; check split dates")
    if not val_mask.any():
        raise ValueError("Validation split is empty; check split dates")
    if not test_mask.any():
        raise ValueError("Test split is empty; check split dates")

    train_features, train_targets = _subset_dataset(dataset, train_mask)
    val_features, val_targets = _subset_dataset(dataset, val_mask)
    test_features, test_targets = _subset_dataset(dataset, test_mask)

    return ArraySplits(
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets,
        test_features=test_features,
        test_targets=test_targets,
        full_features=dataset.features,
        full_targets=dataset.targets,
        target_index=target_index,
    )


def _standardize(
    arrays: ArraySplits,
    method: str,
) -> tuple[ArraySplits, ZScoreScaler | None]:
    if method == "none":
        return arrays, None
    if method != "zscore":
        msg = f"Unsupported standardization method: {method}"
        raise ValueError(msg)

    train_scaled, scaler = ZScoreScaler.fit_transform(arrays.train_features)

    def _transform(data: np.ndarray) -> np.ndarray:
        return scaler.transform(data)

    return (
        ArraySplits(
            train_features=train_scaled,
            train_targets=arrays.train_targets,
            val_features=_transform(arrays.val_features),
            val_targets=arrays.val_targets,
            test_features=_transform(arrays.test_features),
            test_targets=arrays.test_targets,
            full_features=_transform(arrays.full_features),
            full_targets=arrays.full_targets,
            target_index=arrays.target_index,
        ),
        scaler,
    )


def _to_tensor_dataset(features: np.ndarray, targets: np.ndarray) -> TensorDataset:
    feats = torch.from_numpy(features.astype(np.float32, copy=False))
    targs = torch.from_numpy(targets.astype(np.float32, copy=False))
    return TensorDataset(feats, targs)


class TransformerModule(pl.LightningModule):
    """Lightning wrapper orchestrating optimisation."""

    def __init__(
        self,
        model_config: TransformerModelConfig,
        train_config: TrainingConfig,
        input_dim: int,
        seq_len: int,
        target_dim: int,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        itransformer_cfg = model_config.to_itransformer(input_dim, seq_len, target_dim)
        self.model = ITransformer(itransformer_cfg)
        self.loss_fn = nn.MSELoss()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(batch)

    def training_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        features, targets = batch
        preds = self(features)
        loss = self.loss_fn(preds, targets)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        features, targets = batch
        preds = self(features)
        loss = self.loss_fn(preds, targets)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):  # type: ignore[override]
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay,
        )


def _build_dataloaders(arrays: ArraySplits, cfg: TrainingConfig) -> dict[str, DataLoader]:
    train_ds = _to_tensor_dataset(arrays.train_features, arrays.train_targets)
    val_ds = _to_tensor_dataset(arrays.val_features, arrays.val_targets)
    test_ds = _to_tensor_dataset(arrays.test_features, arrays.test_targets)

    return {
        "train": DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False
        ),
        "val": DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=False
        ),
        "test": DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=False
        ),
    }


def _inference(
    module: TransformerModule,
    arrays: ArraySplits,
    batch_size: int,
) -> np.ndarray:
    module.eval()
    device = module.device
    dataset = _to_tensor_dataset(arrays.full_features, arrays.full_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for features, _ in loader:
            outputs = module(features.to(device))
            preds.append(outputs.cpu().numpy())
    return np.concatenate(preds, axis=0)


def _persist_artifacts(
    run_dir: Path,
    forecast_path: Path,
    arrays: ArraySplits,
    predictions: np.ndarray,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    scaler: ZScoreScaler | None,
    metadata_packages: Iterable[str],
    config: TransformerRunConfig,
    best_checkpoint: Path,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    forecast_path.parent.mkdir(parents=True, exist_ok=True)

    # Forecast CSV
    target_values = arrays.full_targets
    if target_values.shape[1] != 1:
        msg = "Forecast export expects a single target column"
        raise ValueError(msg)
    forecast = pd.DataFrame(
        {
            "Date": arrays.target_index,
            "target": target_values[:, 0],
            "yhat": predictions[:, 0],
        }
    )
    forecast["Date"] = forecast["Date"].dt.strftime("%Y-%m-%d")
    forecast.to_csv(forecast_path, index=False)

    checkpoint_dest = run_dir / "checkpoint.ckpt"
    checkpoint_dest.write_bytes(best_checkpoint.read_bytes())

    with (run_dir / "scaler.pkl").open("wb") as file:
        pickle.dump(scaler, file)

    columns_payload = {
        "feature_columns": list(feature_columns),
        "target_columns": list(target_columns),
        "lookback": config.data.lookback,
    }
    (run_dir / "columns.json").write_text(
        json.dumps(columns_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    metadata = collect_run_metadata(seed=config.train.seed, packages=metadata_packages)
    metadata.write_json(run_dir / "metadata.json")

    (run_dir / "resolved_config.json").write_text(
        config.model_dump_json(indent=2),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the iTransformer forecaster.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = TransformerRunConfig.from_yaml(args.config)

    set_global_seed(config.train.seed)
    pl.seed_everything(config.train.seed, workers=True)

    panel, target = load_panels(config.data)
    dataset = make_windows(panel, target, config.data.lookback, config.data.horizon)
    arrays = _split_windows(dataset, config.split)
    arrays, scaler = _standardize(arrays, config.data.standardize)

    dataloaders = _build_dataloaders(arrays, config.train)

    input_dim = panel.shape[1]
    seq_len = config.data.lookback
    target_dim = target.shape[1]

    module = TransformerModule(
        model_config=config.model,
        train_config=config.train,
        input_dim=input_dim,
        seq_len=seq_len,
        target_dim=target_dim,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.output_root / timestamp
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.train.early_stopping_patience,
            mode="min",
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=run_dir,
            filename="checkpoint",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=False,
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=config.train.max_epochs,
        accelerator="auto",
        devices=1,
        deterministic=True,
        default_root_dir=run_dir,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_checkpointing=True,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, train_dataloaders=dataloaders["train"], val_dataloaders=dataloaders["val"])

    checkpoint_cb = callbacks[1]
    if not isinstance(checkpoint_cb, pl.callbacks.ModelCheckpoint):  # pragma: no cover - defensive
        raise RuntimeError("ModelCheckpoint callback misconfigured")
    best_path = Path(checkpoint_cb.best_model_path)
    if not best_path.is_file():
        raise RuntimeError("Best checkpoint not found; training may have failed")

    best_module = TransformerModule.load_from_checkpoint(
        best_path,
        model_config=config.model,
        train_config=config.train,
        input_dim=input_dim,
        seq_len=seq_len,
        target_dim=target_dim,
    )
    best_module.eval()

    predictions = _inference(best_module, arrays, config.train.batch_size)

    _persist_artifacts(
        run_dir=run_dir,
        forecast_path=config.forecast_csv,
        arrays=arrays,
        predictions=predictions,
        feature_columns=list(panel.columns),
        target_columns=list(target.columns),
        scaler=scaler,
        metadata_packages=config.packages,
        config=config,
        best_checkpoint=best_path,
    )

    print(f"Training complete. Artefacts stored at {run_dir}")
    print(f"Forecast saved to {config.forecast_csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
