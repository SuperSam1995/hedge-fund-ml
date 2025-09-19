#!/bin/bash
#SBATCH -J itrafo-train
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH --mem=32G
#SBATCH -o logs/itrafo_%j.out
set -euo pipefail

module purge
# (load your site’s python if needed)
# module load python/3.11

# If your cluster has no internet on workers, pre-sync a wheelhouse or build the venv on a login node first.

uv sync --group dev --group deep-learning-torch
uv run --group dev --group deep-learning-torch python -m scripts.train_transformer --config configs/transformer.yaml
