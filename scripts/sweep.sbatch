#!/bin/bash
#SBATCH --job-name=gps-ec-sweep
#SBATCH --partition=gpu          # CARC: adjust to your GPU partition
#SBATCH --gres=gpu:1             # one GPU per array task
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm-%A_%a.out
# Launch with, e.g.:
#   N=$(wc -l < runs/manifest.txt)
#   sbatch --array=0-$((N-1))%8 scripts/sweep.sbatch   # %8 = 8 concurrent GPUs
#
# Each array task runs ONE line of runs/manifest.txt on its own GPU.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
export WANDB_MODE=offline WANDB_SILENT=true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" runs/manifest.txt)
echo "[task ${SLURM_ARRAY_TASK_ID}] $LINE"
eval "$LINE"
