#!/bin/bash
#SBATCH -J hfml-bundle
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH -t 00:20:00
#SBATCH --mem=4G
#SBATCH -o logs/bundle_%j.out
set -euo pipefail
uv sync --group dev
make bundle
