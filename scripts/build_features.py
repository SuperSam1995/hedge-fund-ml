"""Config-driven feature construction CLI."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from pipeline.features import build_features, load_feature_config, persist_artifacts

DEFAULT_CONFIG_PATH = Path("configs/features.yaml")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build engineered features from ETF and hedge fund data",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the feature configuration YAML file.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_feature_config(args.config)
    artifacts = build_features(config)
    persist_artifacts(config, artifacts)
    print(f"Features saved to {config.data.output_features}")
    print(f"HK span model saved to {config.data.output_model}")


if __name__ == "__main__":
    main()
