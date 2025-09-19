HPC Runbook (Slurm)
===================

Python & env
------------

# Choose Python 3.11 for training
uv python install 3.11
uv venv --python 3.11
uv sync --group dev --group deep-learning-torch


CPU job (safe default) — save as jobs/train_itrafo_cpu.sbatch
------------------------------------------------------------

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


GPU job (optional) — jobs/train_itrafo_gpu.sbatch
-------------------------------------------------

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


Replication + eval (CPU) — jobs/replicate_eval.sbatch
-----------------------------------------------------

#!/bin/bash
#SBATCH -J itrafo-repl-eval
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH --mem=8G
#SBATCH -o logs/itrafo_repl_%j.out
set -euo pipefail

uv sync --group dev
make replicate_itrafo
make eval_itrafo
make report


Submit
------

mkdir -p logs
sbatch jobs/train_itrafo_cpu.sbatch
# then
sbatch jobs/replicate_eval.sbatch


Data & outputs on HPC
---------------------

Put input CSVs under $SCRATCH/hedge-fund-ml/cleaned_data/… or a project volume.

Set paths in configs/*.yaml to point to that location (or symlink).

Artifacts:

models/itrafo/<timestamp>/…

data/interim/itrafo_forecast.csv

reports/metrics/itrafo_metrics.*, reports/final_report.html


Repro tips
----------

Fix seed in configs for fair comparisons.

Archive the exact config and git commit hash with the checkpoint (trainer should save both).


Mac quick-run (developer laptop)
--------------------------------

# 3.11 env for training; 3.13 fine for baseline only
uv python install 3.11
uv sync --group dev --group deep-learning-torch
make train_itrafo
make report_itrafo
open reports/final_report.html
