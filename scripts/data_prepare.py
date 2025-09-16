"""Command-line entry point for the data preparation pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from data.prepare import (
    DataPrepConfig,
    PreparedData,
    align_monthly,
    clean,
    load_raw,
    save_prepared,
)

DEFAULT_CONFIG_PATH = Path("configs/data.yaml")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare monthly hedge fund datasets from raw registry inputs.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration describing registry sources.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and transform data without writing artefacts to disk.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Emit debug-level logs during processing.",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _prepare(config_path: Path) -> tuple[PreparedData, DataPrepConfig]:
    config = DataPrepConfig.from_yaml(config_path)
    raw = load_raw(config)
    cleaned = clean(raw, config)
    aligned = align_monthly(cleaned)
    return aligned, config


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    logging.info("Preparing datasets using %s", args.config)

    prepared, config = _prepare(args.config)
    observations, funds = prepared.hedge_funds.shape
    factors = prepared.factor_etf.shape[1]

    if args.dry_run:
        logging.info(
            "Dry run complete: %s rows, %s hedge funds, %s factor ETFs",
            observations,
            funds,
            factors,
        )
    else:
        save_prepared(prepared, config)
        logging.info(
            "Saved %s rows, %s hedge funds, %s factor ETFs to configured outputs",
            observations,
            funds,
            factors,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
