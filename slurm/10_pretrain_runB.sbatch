#!/bin/bash
# Phase I Step 1 -- source-domain pretraining, one array task per fold.
#   sbatch slurm/10_pretrain.sbatch                 # all 5 folds
#   sbatch --array=0 slurm/10_pretrain.sbatch       # just fold 0
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=18:00:00      # measured 12.7 h/fold at batches_per_epoch=20000
#SBATCH --constraint=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-4
#SBATCH -J gmtsB_pre
#SBATCH -o logs/pretrainB_%A_%a.out
#SBATCH -e logs/pretrainB_%A_%a.err

set -eo pipefail
# SLURM copies this script to a node-local spool directory, so BASH_SOURCE does
# NOT point at the repo. Use the submit directory, or GMTS_REPO if you submit from
# elsewhere, and verify it really is the repo before touching anything.
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$REPO_ROOT/configs/phase1.yaml" ]]; then
  echo "REPO_ROOT=$REPO_ROOT is not the global_mtslstm repo (no configs/phase1.yaml)." >&2
  echo "Submit from the repo root, or export GMTS_REPO=/path/to/global_mtslstm." >&2
  exit 2
fi
cd "$REPO_ROOT"
mkdir -p logs

source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

FOLD="${SLURM_ARRAY_TASK_ID:-0}"
# W&B logs live outside $HOME: it sits at 191 G against a 180 G soft quota.
export WANDB_DIR="${WANDB_DIR:-/ibex/user/kongw0a/global_mtslstm_wandb}"
mkdir -p "$WANDB_DIR"
# Say plainly whether live monitoring will work. ibex compute nodes DO reach
# api.wandb.ai; without a key the run silently drops to offline and you would
# only find out when nothing appears in the dashboard.
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "W&B: key present, online monitoring enabled (dir $WANDB_DIR)"
else
  echo "W&B: WANDB_API_KEY is NOT set -> the run will fall back to OFFLINE." >&2
  echo "     Live monitoring will not work. Fix with 'export WANDB_API_KEY=...'" >&2
  echo "     in ~/.bashrc, or sync afterwards with 'wandb sync $WANDB_DIR/offline-run-*'." >&2
fi

echo "fold=$FOLD python=$PYTHON_BIN node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PYTHON_BIN" -u -m train.pretrain_source --config configs/phase1_runB.yaml \
  --fold "$FOLD" \
  --resume \
  --set "wandb.group=phase1B_pretrain" ${EXTRA_SET[@]+"${EXTRA_SET[@]}"}
