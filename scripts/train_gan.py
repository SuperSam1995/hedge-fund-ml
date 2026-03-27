"""Train the Wasserstein GAN using configuration-driven inputs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, ValidationError

from hedge_fund_ml import set_global_seed
from hedge_fund_ml.models import WGANConfig, WGANOutputConfig, train_gan

DEFAULT_CONFIG_PATH = Path("configs/wgan.yaml")


class DataSourceConfig(BaseModel):
    path: Path
    index_col: str | None = "Date"
    parse_dates: bool = True
    columns: list[str] | None = None

    model_config = {"extra": "forbid"}

    def load(self) -> pd.DataFrame:
        if self.parse_dates and self.index_col is not None:
            frame = pd.read_csv(
                self.path,
                index_col=self.index_col,
                parse_dates=[self.index_col],
            )
        else:
            frame = pd.read_csv(self.path, index_col=self.index_col)
        if self.columns is not None:
            missing = set(self.columns) - set(frame.columns)
            if missing:
                raise ValueError(f"Missing columns in {self.path}: {sorted(missing)}")
            frame = frame.loc[:, self.columns]
        return frame.sort_index()


class WGANDataConfig(BaseModel):
    primary: DataSourceConfig
    secondary: DataSourceConfig | None = None
    window: int = Field(default=24, gt=1)
    samples: int = Field(default=256, gt=0)
    dropna: bool = True
    shuffle: bool = True

    model_config = {"extra": "forbid"}

    def load(self, seed: int) -> np.ndarray:
        frame = self.primary.load()
        if self.secondary is not None:
            other = self.secondary.load()
            frame = frame.join(other, how="inner")
        if self.dropna:
            frame = frame.dropna(axis=0, how="any")
        if len(frame) < self.window:
            raise ValueError("Not enough rows to construct sliding windows")
        array = frame.to_numpy(dtype=np.float32)
        total_windows = len(frame) - self.window + 1
        indices = np.arange(total_windows)
        if self.shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        limit = min(self.samples, total_windows)
        selected = indices[:limit]
        windows = np.stack([array[idx : idx + self.window] for idx in selected])
        return windows


def _load_config(path: Path) -> tuple[WGANConfig, WGANDataConfig]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data_payload = payload.pop("data", None)
    if data_payload is None:
        raise ValueError("Configuration missing data section")
    gan_cfg = WGANConfig.model_validate(payload)
    data_cfg = WGANDataConfig.model_validate(data_payload)
    return gan_cfg, data_cfg


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Wasserstein GAN.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        gan_cfg, data_cfg = _load_config(args.config)
    except FileNotFoundError as exc:
        raise SystemExit(f"Config file not found: {args.config}") from exc
    except (ValidationError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    set_global_seed(gan_cfg.seed)
    data = data_cfg.load(gan_cfg.seed)

    artifacts = train_gan(data, gan_cfg)
    output_cfg = WGANOutputConfig(root=gan_cfg.output.root, run_path=artifacts.run_dir)
    resolved = gan_cfg.model_copy(update={"output": output_cfg})
    (artifacts.run_dir / "resolved_config.yaml").write_text(
        resolved.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print(f"WGAN training complete. Artefacts -> {artifacts.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
