# HPC Runbook (Slurm)

## Environment (CPU-only baseline)
```bash
uv venv --python 3.13
uv sync --group dev
```

## Submit baseline pipeline
```bash
mkdir -p logs
sbatch jobs/reproduce_cpu.sbatch
```

## Monitor
```bash
squeue -u $USER
tail -f logs/reproduce_*.out
```

## Inputs & Outputs

- **Inputs** (repo-relative): `cleaned_data/factor_etf_data.csv`, `cleaned_data/hfd.csv`
- **Outputs**: `data/interim/*`, `results/metrics/*`, `results/figures/*`, `reports/final_report.html`

## Run locally (smoke test)
```bash
uv sync --group dev
make reproduce_cpu
```

## LLM bundle on HPC
```bash
sbatch jobs/bundle_cpu.sbatch
```

**Done when**
- `sbatch jobs/bundle_cpu.sbatch` creates a new `results/logs/bundle/<run_id>/` on the cluster.
