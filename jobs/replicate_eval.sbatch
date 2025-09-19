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
