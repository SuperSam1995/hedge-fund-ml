#!/bin/bash
#SBATCH -J hfml-reproduce
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH --mem=8G
#SBATCH -o logs/reproduce_%j.out
set -euo pipefail

python -m scripts.print_env || true
uv sync --group dev
uv run --group dev python -m scripts.build_features --config configs/features.yaml
uv run --group dev python -m scripts.replicate     --config configs/replicate.yaml
uv run --group dev python -m scripts.eval          --config configs/eval.yaml
# ensure dirs exist
mkdir -p reports results results/figures
# convert text notebook → ipynb (always fresh)
uv run --group dev jupytext --to ipynb notebooks/final_report.py -o reports/_tmp.ipynb

uv run --group dev papermill reports/_tmp.ipynb reports/_exec.ipynb \
  -p metrics_path results/metrics/replication_metrics.json \
  -p figures_dir results/figures
uv run --group dev jupyter nbconvert --to html --no-input \
  --output-dir reports --output final_report.html reports/_exec.ipynb
echo "Report: reports/final_report.html"
