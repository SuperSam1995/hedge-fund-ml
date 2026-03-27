"""CLI entry-point for the replication pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from pipeline.replicate import ReplicateConfig, run_replication

DEFAULT_CONFIG_PATH = Path("configs/replicate.yaml")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the replication pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the replication YAML configuration.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = ReplicateConfig.from_yaml(args.config)
    result = run_replication(config)
    print("Replication complete")
    print(f"Features: {result.feature_frame}")
    print(f"HK model: {result.hk_model_path}")
    print(f"Weights: {result.weights_path}")
    print(f"Scaler: {result.scaler_path}")


if __name__ == "__main__":
    main()
