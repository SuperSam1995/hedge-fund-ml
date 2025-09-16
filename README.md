# Hedge Fund ML Research Toolkit

Modernised research tooling for the "Do You Really Need to Pay 2/20?" hedge
fund replication study.  The repository now packages the reusable analytics,
locks dependencies, and provides continuous integration friendly workflows.

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
├── requirements.lock       # Runtime lockfile (pip-compile managed)
└── requirements-dev.lock   # Development lockfile with lint/test tooling
```

## Environment management

1. Install dependencies from the compiled lockfile:
   ```bash
   pip install -r requirements-dev.lock
   ```
2. Refresh the lockfiles after dependency updates:
   ```bash
   pip-compile --resolver=backtracking --output-file requirements.lock pyproject.toml
   pip-compile --resolver=backtracking --extra dev --output-file requirements-dev.lock pyproject.toml
   ```
3. Install deep-learning extras only when working with the autoencoder/GAN components:
   ```bash
   pip install -e .[deep-learning]
   ```

The project targets Python 3.10+ and keeps notebooks data-only by consuming the
`hedge_fund_ml` package.

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
metadata.write_json("reports/run_metadata.json")
```

The seeding utility touches Python, NumPy, PyTorch (if installed), TensorFlow
(if installed) and toggles deterministic CuDNN behaviour.  Metadata logging
captures package versions together with the Git commit hash.

## Reporting

Matplotlib visualisations should be single-purpose figures with labelled axes
and saved into `reports/figures/` for reproducibility.

## Quality gates

All checks run through `pre-commit` and CI:

```bash
pre-commit install
pre-commit run --all-files
pytest -q
```

CI is defined in `.github/workflows/ci.yml` and enforces ruff, black, mypy and
the test suite.
