"""Command-line interface for training the autoencoder."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from pydantic import ValidationError
from sklearn.model_selection import train_test_split

from hedge_fund_ml import set_global_seed
from hedge_fund_ml.models import AutoencoderConfig, AutoencoderOutputConfig, fit

DEFAULT_CONFIG_PATH = Path("configs/ae.yaml")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the factor autoencoder on prepared ETF features.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    return parser


def _load_config(path: Path) -> AutoencoderConfig:
    try:
        return AutoencoderConfig.from_yaml(path)
    except FileNotFoundError as exc:
        raise SystemExit(f"Config file not found: {path}") from exc
    except ValidationError as exc:
        raise SystemExit(str(exc)) from exc


def _split_data(
    frame: pd.DataFrame, cfg: AutoencoderConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_cfg = cfg.data
    if data_cfg is None:
        raise SystemExit("AutoencoderConfig.data must be defined for training")
    if data_cfg.dropna:
        frame = frame.dropna(axis=0, how="any")
    fraction = data_cfg.validation_fraction
    if fraction <= 0.0 or fraction >= 1.0:
        raise SystemExit("validation_fraction must be between 0 and 1")
    if data_cfg.shuffle:
        train, val = train_test_split(
            frame,
            test_size=fraction,
            shuffle=True,
            random_state=cfg.seed,
        )
    else:
        split_index = int(len(frame) * (1 - fraction))
        if split_index <= 0 or split_index >= len(frame):
            raise SystemExit("validation split results in empty train/val sets")
        train = frame.iloc[:split_index]
        val = frame.iloc[split_index:]
    return train, val


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    if config.data is None:
        raise SystemExit("Configuration missing data section")

    set_global_seed(config.seed)

    features = pd.read_csv(
        config.data.features_path,
        index_col=0,
        parse_dates=True,
    )
    train, val = _split_data(features, config)

    result = fit(train, val, config)
    output_cfg = AutoencoderOutputConfig(
        root=config.output.root, run_path=result.run_dir
    )
    summary = result.metrics.set_index("split")
    print(f"Autoencoder training complete. Artefacts -> {result.run_dir}")
    print(summary)

    # Update configuration snapshot with the concrete run directory for convenience.
    updated_cfg = config.model_copy(update={"output": output_cfg})
    (result.run_dir / "resolved_config.yaml").write_text(
        updated_cfg.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
