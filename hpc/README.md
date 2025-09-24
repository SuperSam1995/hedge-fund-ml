# HPC Readiness Guide

This document outlines how to launch the training and evaluation workflows on a SLURM-based high-performance computing (HPC) cluster while keeping all generated artifacts in the `results/` tree and ensuring the repository stays free of binary blobs.

## 1. Example SLURM batch scripts

### 1.1 Training job (`jobs/train.slurm`)
```bash
#!/bin/bash
#SBATCH --job-name=hf-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

module purge
module load cuda/12.1 python/3.10

source ~/venvs/hedge-fund-ml/bin/activate

export HF_RUN_MODE=train
python -m scripts.train \
  --config configs/training/default.yaml \
  --output-dir results/experiments/
```

### 1.2 Evaluation job (`jobs/eval.slurm`)
```bash
#!/bin/bash
#SBATCH --job-name=hf-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

module purge
module load cuda/12.1 python/3.10

source ~/venvs/hedge-fund-ml/bin/activate

python -m scripts.evaluate \
  --checkpoint results/experiments/run_001/weights.pt \
  --config configs/eval/default.yaml \
  --output-dir results/evaluations/run_001/
```

> **Why this design?** Each script purges and reloads modules for deterministic environments, activates a pre-created virtual environment, and writes stdout/stderr plus artifacts into subfolders of `results/` so raw data and code remain untouched.

## 2. Directing outputs to `results/`

1. Create structured subdirectories before launching jobs:
   ```bash
   mkdir -p results/{logs,experiments,evaluations,metadata}
   ```
2. Always pass `--output-dir` arguments under `results/` when invoking `scripts.train`, `scripts.evaluate`, or related entry points.
3. Configure logging (e.g., via `configs/training/default.yaml`) to write checkpoints, metrics, and plots inside the provided `--output-dir`.
4. Store immutable run metadata (software versions, git commit hash, RNG seeds) in `results/metadata/` to preserve reproducibility without polluting Git history.

## 3. Step-by-step HPC workflow

1. **Log in to the cluster** via SSH and switch to the project directory:
   ```bash
   ssh user@cluster
   cd /path/to/hedge-fund-ml
   git pull --ff-only
   ```
2. **Prepare the software environment**:
   ```bash
   module purge
   module load cuda/12.1 python/3.10 git
   python -m venv ~/venvs/hedge-fund-ml
   source ~/venvs/hedge-fund-ml/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt  # or `uv sync` if available
   pre-commit install
   ```
3. **Create results directories** and ensure raw data stay immutable:
   ```bash
   mkdir -p results/{logs,experiments,evaluations,metadata}
   chmod -R u+w results
   ```
4. **Submit training job**:
   ```bash
   sbatch jobs/train.slurm
   ```
5. **Monitor progress**:
   ```bash
   squeue --me
   tail -f results/logs/hf-train-<jobid>.out
   ```
6. **Run evaluation** (after training completion):
   ```bash
   sbatch --dependency=afterok:<train_jobid> jobs/eval.slurm
   ```
7. **Collect artifacts** by syncing the `results/` subtree back to local storage or long-term object storage. Do **not** commit binary checkpoints; track them with DVC or a similar registry if needed.
8. **Clean up** temporary scratch files while leaving `data/raw/` untouched and never pushing binary blobs to Git.

## 4. Repository hygiene

- Only text-based assets (YAML configs, scripts, and documentation) belong in Git.
- Keep checkpoints, NumPy arrays, and other binaries strictly in `results/` or external storage.
- Use `.gitignore` to prevent accidental commits of `results/` contents.

Following this process keeps the codebase reproducible, transparent, and ready for high-throughput HPC execution without polluting version control.
