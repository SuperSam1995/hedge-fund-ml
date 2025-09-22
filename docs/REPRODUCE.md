# Reproduction workflow

## Command overview

Ensure dependencies are installed once via:

```bash
uv sync --group dev --extra deep-learning
```

Thereafter run the entire research pipeline, including optional deep-learning
components, with:

```bash
make reproduce
```

The Makefile target invokes `uv run --group dev --extra deep-learning -m
hedge_fund_ml.cli reproduce`, ensuring the TensorFlow/Keras extras are available
for the GAN and autoencoder placeholders while respecting the project lockfile.
`uv run` shells into the managed virtual environment, so there is no need to
activate `.venv/` manually.

## Data expectations

- All datasets referenced in `configs/data_registry.yaml` must exist under the
  `data/` root, respecting the `raw/`, `interim/` and `processed/` layout.
- Raw assets remain immutable; verification hashes defined in the registry are
  checked before any downstream step executes.
- Derived datasets should be written into `data/interim/` or
  `data/processed/`, keeping the `raw/` subtree untouched for reproducibility.

## Artefact destinations

Running `make reproduce` produces deterministic outputs in the following
locations:

- Stage-level run metadata JSON documents in `results/logs/`.
- Replication payload persisted to `results/logs/replication.json`.
- Evaluation metrics snapshot at `results/metrics/metrics.json`.
- Matplotlib figure saved to `results/figures/replication_placeholder.png` with
  labelled axes.

These artefacts, together with the seed fixes declared in `configs/run.yaml`,
facilitate reproducible benchmarking runs.
