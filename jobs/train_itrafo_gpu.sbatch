#!/bin/bash
#SBATCH -J itrafo-train-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 02:00:00
#SBATCH --mem=32G
#SBATCH -o logs/itrafo_gpu_%j.out
set -euo pipefail

module purge
# module load cuda/12.1  # if your site requires
uv python install 3.11
uv sync --group dev --group deep-learning-torch

# PyTorch will auto-pick GPU; if you need to force: export CUDA_VISIBLE_DEVICES=0
uv run --group dev --group deep-learning-torch python -m scripts.train_transformer --config configs/transformer.yaml
