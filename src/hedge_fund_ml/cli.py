"""Command-line helpers orchestrating the research pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import yaml
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field, ValidationError

from hedge_fund_ml import DataRegistry, collect_run_metadata, set_global_seed

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_REGISTRY_PATH = Path("configs/data_registry.yaml")
DEFAULT_RUN_CONFIG_PATH = Path("configs/run.yaml")
DEFAULT_METADATA_DIR = Path("results/logs")
DEFAULT_FIGURE_DIR = Path("results/figures")
DEFAULT_METRICS_PATH = Path("results/metrics/metrics.json")
DEFAULT_PACKAGES: list[str] = ["numpy", "pandas", "scikit-learn", "matplotlib"]
DEEP_LEARNING_EXTRAS = ["tensorflow", "keras"]


class RunConfig(BaseModel):
    """Minimal configuration parsed from YAML for reproducible runs."""

    seed: int = 42
    packages: list[str] = Field(default_factory=lambda: list(DEFAULT_PACKAGES))

    model_config = {"extra": "forbid"}


def _read_yaml(path: Path) -> dict:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"Invalid YAML in {path}: {exc}") from exc
    return payload or {}


def load_run_config(path: Path | None) -> RunConfig:
    target = path or DEFAULT_RUN_CONFIG_PATH
    payload = _read_yaml(target)
    try:
        return RunConfig.model_validate(payload)
    except ValidationError as exc:
        raise SystemExit(str(exc)) from exc


def _log_run(name: str, config: RunConfig, extra_packages: Iterable[str]) -> Path:
    packages = list(dict.fromkeys([*config.packages, *extra_packages]))
    metadata = collect_run_metadata(seed=config.seed, packages=packages)
    DEFAULT_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DEFAULT_METADATA_DIR / f"{name}.json"
    metadata.write_json(destination)
    print(f"[{name}] metadata -> {destination}")
    return destination


def run_data(
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> None:
    registry = DataRegistry.from_yaml(data_root, registry_path)
    print(f"Validating datasets listed in {registry_path} against {data_root}")
    for dataset in registry.datasets():
        location = registry.path(dataset)
        try:
            verified = registry.verify(dataset)
        except FileNotFoundError:
            verified = False
        status = "verified" if verified else "missing"
        print(f"- {dataset:<20} {status:<9} {location}")


def run_train(
    name: str,
    config_path: Path | None,
    extra_packages: Iterable[str],
) -> Path:
    config = load_run_config(config_path)
    seeded = set_global_seed(config.seed)
    print(f"[{name}] seeds -> {seeded}")
    return _log_run(name, config, extra_packages)


def run_autoencoder(config_path: Path | None) -> Path:
    return run_train("train_autoencoder", config_path, DEEP_LEARNING_EXTRAS)


def run_gan(config_path: Path | None) -> Path:
    return run_train("train_gan", config_path, DEEP_LEARNING_EXTRAS)


def run_replicate(
    config_path: Path | None,
    output_path: Path = Path("results/logs/replication.json"),
) -> Path:
    config = load_run_config(config_path)
    seeded = set_global_seed(config.seed)
    print(f"[replicate] seeds -> {seeded}")
    _log_run("replicate", config, [])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": config.seed,
        "packages": config.packages,
        "note": "Synthetic replication artefact placeholder.",
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[replicate] artefact -> {output_path}")
    return output_path


def run_eval(
    config_path: Path | None,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> Path:
    config = load_run_config(config_path)
    seeded = set_global_seed(config.seed)
    print(f"[eval] seeds -> {seeded}")
    _log_run("evaluate", config, [])
    rng = np.random.default_rng(config.seed)
    metrics = {
        "seed": config.seed,
        "mean_return": float(rng.normal(0.01, 0.001)),
        "sharpe": float(rng.normal(1.0, 0.1)),
        "max_drawdown": float(rng.uniform(0.05, 0.15)),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[eval] metrics -> {metrics_path}")
    return metrics_path


def run_report(
    config_path: Path | None,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    figure_path: Path = DEFAULT_FIGURE_DIR / "replication_placeholder.png",
) -> Path:
    config = load_run_config(config_path)
    seeded = set_global_seed(config.seed)
    print(f"[report] seeds -> {seeded}")
    _log_run("report", config, [])
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        mean_return = float(metrics.get("mean_return", 0.0))
        sharpe = float(metrics.get("sharpe", 0.0))
    else:
        mean_return = 0.0
        sharpe = 0.0
    x = np.linspace(0.0, 1.0, 50)
    y = mean_return + sharpe * (x - 0.5)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, linewidth=2.0)
    ax.set_xlabel("Normalised time")
    ax.set_ylabel("Estimated return")
    ax.set_title("Replication performance (synthetic)")
    ax.grid(True, linestyle="--", alpha=0.3)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[report] figure -> {figure_path}")
    return figure_path


def run_reproduce(
    config_path: Path | None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> None:
    run_data(registry_path=registry_path, data_root=data_root)
    run_autoencoder(config_path)
    run_gan(config_path)
    run_replicate(config_path)
    run_eval(config_path)
    run_report(config_path)
    print("[reproduce] pipeline complete")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research workflow helper CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_argument(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--config",
            type=Path,
            default=None,
            help=("Path to run configuration YAML " f"(default: {DEFAULT_RUN_CONFIG_PATH})."),
        )

    def add_data_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--registry",
            type=Path,
            default=DEFAULT_REGISTRY_PATH,
            help="Path to data registry YAML.",
        )
        command.add_argument(
            "--data-root",
            type=Path,
            default=DEFAULT_DATA_ROOT,
            help="Root directory for project datasets.",
        )

    data_parser = subparsers.add_parser("data", help="Validate datasets registered in YAML")
    add_data_arguments(data_parser)

    def data_cmd(args: argparse.Namespace) -> None:
        run_data(registry_path=args.registry, data_root=args.data_root)

    data_parser.set_defaults(func=data_cmd)

    train_ae_parser = subparsers.add_parser("train-ae", help="Run the autoencoder training stub")
    add_config_argument(train_ae_parser)

    def train_ae_cmd(args: argparse.Namespace) -> None:
        run_autoencoder(args.config)

    train_ae_parser.set_defaults(func=train_ae_cmd)

    train_gan_parser = subparsers.add_parser("train-gan", help="Run the GAN training stub")
    add_config_argument(train_gan_parser)

    def train_gan_cmd(args: argparse.Namespace) -> None:
        run_gan(args.config)

    train_gan_parser.set_defaults(func=train_gan_cmd)

    replicate_parser = subparsers.add_parser("replicate", help="Generate replication artefacts")
    add_config_argument(replicate_parser)

    def replicate_cmd(args: argparse.Namespace) -> None:
        run_replicate(args.config)

    replicate_parser.set_defaults(func=replicate_cmd)

    eval_parser = subparsers.add_parser("eval", help="Evaluate model outputs")
    add_config_argument(eval_parser)

    def eval_cmd(args: argparse.Namespace) -> None:
        run_eval(args.config)

    eval_parser.set_defaults(func=eval_cmd)

    report_parser = subparsers.add_parser("report", help="Create reporting artefacts")
    add_config_argument(report_parser)

    def report_cmd(args: argparse.Namespace) -> None:
        run_report(args.config)

    report_parser.set_defaults(func=report_cmd)

    reproduce_parser = subparsers.add_parser("reproduce", help="Execute the full pipeline")
    add_config_argument(reproduce_parser)
    add_data_arguments(reproduce_parser)

    def reproduce_cmd(args: argparse.Namespace) -> None:
        run_reproduce(
            config_path=args.config,
            registry_path=args.registry,
            data_root=args.data_root,
        )

    reproduce_parser.set_defaults(func=reproduce_cmd)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
