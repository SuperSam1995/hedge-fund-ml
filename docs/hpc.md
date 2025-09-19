# HPC Runbook (Slurm)

## Python / env
Baseline (CPU-only) works on Python **3.13+**:
```bash
uv venv --python 3.13
uv sync --group dev
```

If you later train deep models (PyTorch), prefer 3.11 on HPC:
```bash
uv python install 3.11
uv venv --python 3.11
uv sync --group dev --group deep-learning-torch
```

Submit baseline pipeline (CPU)
mkdir -p logs
sbatch jobs/reproduce_cpu.sbatch

Monitor jobs
squeue -u $USER
tail -f logs/*.out

Inputs & outputs

Inputs: cleaned_data/factor_etf_data.csv, cleaned_data/hfd.csv (relative to repo).

Outputs:

data/interim/*.csv (features/weights)

reports/metrics/*.json|csv, reports/figures/*.png

Final HTML: reports/final_report.html
