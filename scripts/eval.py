"""CLI entry-point for the evaluation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from pipeline.evaluate import EvaluationConfig, run_evaluation

DEFAULT_CONFIG_PATH = Path("configs/eval.yaml")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate replication outputs")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the evaluation YAML configuration.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = EvaluationConfig.from_yaml(args.config)
    result = run_evaluation(config)
    print("Evaluation complete")
    print(result.metrics)


if __name__ == "__main__":
    main()
