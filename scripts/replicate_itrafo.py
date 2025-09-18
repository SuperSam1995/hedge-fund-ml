"""CLI entry-point for the iTraFo replication decoder."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from replicate import ITrafoConfig, run_itrafo_replication

DEFAULT_CONFIG_PATH = Path("configs/replicate_itrafo.yaml")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the iTraFo replication decoder")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the decoder YAML configuration.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = ITrafoConfig.from_yaml(args.config)
    result = run_itrafo_replication(config)
    print("iTraFo replication complete")
    print(f"Weights saved to: {config.paths.weights_csv}")
    print(f"Series saved to: {config.paths.series_csv}")
    if result.metadata_path is not None:
        print(f"Metadata saved to: {result.metadata_path}")


if __name__ == "__main__":
    main()
