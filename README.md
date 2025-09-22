# Hedge Fund ML Research Toolkit

Modernised research tooling for the "Do You Really Need to Pay 2/20?" hedge
fund replication study.  The repository now packages the reusable analytics,
locks dependencies, and provides continuous integration friendly workflows.

Recommended: Python 3.11 for deep learning.

Baseline pipeline works on 3.13; training jobs use 3.11.

## Repository layout

```
├── configs/                # YAML configuration managed through Pydantic
├── data/
│   ├── raw/                # Immutable inputs registered via DataRegistry
│   ├── interim/            # Deterministic intermediate artefacts
│   └── processed/          # Final, model-ready datasets
├── src/hedge_fund_ml/      # Importable Python package
│   ├── data/               # Data registry utilities
│   ├── models/             # ML models (e.g. autoencoder encapsulation)
│   ├── telemetry/          # Run metadata helpers
│   └── utils/              # Finance utilities and seeding helpers
├── tests/                  # Pytest suite guarding the helpers and registry
└── uv.lock                 # Unified uv lockfile with runtime + dev groups
```

## Environment management

1. Install dependencies with [uv](https://docs.astral.sh/uv/):
   ```bash
   uv python install 3.11
   uv sync --group dev --group deep-learning-torch
   ```
   This provisions the in-repo `.venv/` with both runtime and deep learning tooling on
   Python 3.11.
2. Refresh the lockfile after dependency updates:
   ```bash
   uv lock --upgrade-package <name>
   ```
   Use `uv lock` (without arguments) for a full refresh. Commit the resulting `uv.lock` to
   keep CI reproducible.
3. CI keeps running against the baseline configuration (Python 3.13, runtime groups only)
   to ensure the classical pipeline remains green. GPU-oriented workflows should target
   Python 3.11 locally or on HPC nodes.

The project targets Python 3.10+ and keeps notebooks data-only by consuming the
`hedge_fund_ml` package.

### Running commands via `uv`

No manual environment activation is required. Use `uv run` to execute tooling
inside the locked environment, for example:

```bash
uv run --group dev pytest -q
uv run --group dev --group deep-learning-torch python -m hedge_fund_ml.cli reproduce
```

This guarantees the package (and optional deep learning group) import cleanly during tests
or when running the CLI.

## Data governance

Data lives under `data/` and is tracked via `configs/data_registry.yaml`.  The
`DataRegistry` class validates the YAML entries, ensures stage directories
(`raw`, `interim`, `processed`) exist, and can verify checksums for immutable
raw files.

```python
from hedge_fund_ml import DataRegistry, RegistryConfig

registry = DataRegistry.from_yaml("data", "configs/data_registry.yaml")
path = registry.path("etf_data")
assert registry.verify("etf_data"), "raw dataset hash mismatch"
```

## Reproducibility helpers

```python
from hedge_fund_ml import set_global_seed, collect_run_metadata

seed_info = set_global_seed(42)
metadata = collect_run_metadata(seed=42, packages=["numpy", "pandas", "scikit-learn"])
metadata.write_json("results/logs/run_metadata.json")
```

The seeding utility touches Python, NumPy, PyTorch (if installed), TensorFlow
(if installed) and toggles deterministic CuDNN behaviour.  Metadata logging
captures package versions together with the Git commit hash.

## End-to-end pipeline (`make reproduce`)

Run the full deterministic workflow — registry validation, model stubs,
replication, evaluation and reporting — via:

```bash
make reproduce
```

The target shells out to `uv run --group dev --group deep-learning-torch` so that the
optional PyTorch Lightning/Optuna dependencies required by the GAN/autoencoder shims
are available. Expect the command to:

1. Validate that all datasets declared in `configs/data_registry.yaml` exist
   under `data/{raw,interim,processed}` and that raw files hash-match their
   catalogued values.
2. Honour the YAML run configuration at `configs/run.yaml` (or an override via
   `--config`) for seeding and package metadata capture.
3. Leave artefacts in deterministic locations:
   - Run metadata JSONs inside `results/logs/` for each pipeline stage.
   - Synthetic replication payload at `results/logs/replication.json`.
   - Evaluation metrics snapshot at `results/metrics/metrics.json`.
   - Placeholder figure written to `results/figures/replication_placeholder.png`.

The run assumes immutable raw inputs in `data/raw/` (managed through the
registry) and writes only to `data/interim/` or below `results/` to keep raw
data untouched.

## Reporting

Matplotlib visualisations should be single-purpose figures with labelled axes
and saved into `results/figures/` for reproducibility.

## Quality gates

All checks run through `pre-commit` and CI:

```bash
pre-commit install
pre-commit run --all-files
pytest -q
```

CI is defined in `.github/workflows/ci.yml` and enforces ruff, black, mypy and
the test suite.
