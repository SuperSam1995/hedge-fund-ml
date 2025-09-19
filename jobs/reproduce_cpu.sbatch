#!/bin/bash
#SBATCH -J hfml-reproduce
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH --mem=8G
#SBATCH -o logs/reproduce_%j.out
set -euo pipefail

echo "[ENV]"
python -m scripts.print_env || true

# Build env (uses the repo's pyproject)
uv sync --group dev

# 1) features → 2) replicate → 3) eval
uv run --group dev python -m scripts.build_features --config configs/features.yaml
uv run --group dev python -m scripts.replicate     --config configs/replicate.yaml
uv run --group dev python -m scripts.eval          --config configs/eval.yaml

# 4) render HTML from notebooks/final_report.ipynb
uv run --group dev papermill notebooks/final_report.ipynb reports/_tmp.ipynb \
  -p metrics_path reports/metrics/replication_metrics.json \
  -p figures_dir reports/figures
uv run --group dev jupyter nbconvert --to html --no-input \
  --output-dir reports --output final_report.html reports/_tmp.ipynb

echo "Report: reports/final_report.html"
