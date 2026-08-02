#!/bin/bash
# Phase I Steps 2-3 -- daily-only target transfer + source-degradation check.
# Needs the matching fold's pretrain output to exist.
#   sbatch --dependency=afterok:<pretrain_jobid> slurm/20_transfer.sbatch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --constraint=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-4
#SBATCH -J gmts_xfer
#SBATCH -o logs/transfer_%A_%a.out
#SBATCH -e logs/transfer_%A_%a.err

set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

FOLD="${SLURM_ARRAY_TASK_ID:-0}"
export WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/wandb}"
mkdir -p "$WANDB_DIR"

echo "fold=$FOLD python=$PYTHON_BIN node=$(hostname)"

"$PYTHON_BIN" -u -m train.transfer_target \
  --fold "$FOLD" \
  --peek-hourly \
  --set "wandb.group=phase1_transfer" "${EXTRA_SET[@]}"
