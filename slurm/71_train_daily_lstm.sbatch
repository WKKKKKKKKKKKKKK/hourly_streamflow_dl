#!/bin/bash
# An ordinary daily LSTM on one design's target folds, trained from scratch.
#   CONFIG=configs/phase1_dailylstm_aligned_random.yaml sbatch slurm/71_train_daily_lstm.sbatch
# Continental has four folds:
#   CONFIG=configs/phase1_dailylstm_aligned_continental.yaml sbatch --array=0-3 slurm/71_train_daily_lstm.sbatch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00     # 80 epochs x ~162 s plus the eval passes; patience usually stops earlier
#SBATCH --constraint="v100|a100"
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH -J gmts_dLSTM
#SBATCH -o logs/dailylstm_%A_%a.out
#SBATCH -e logs/dailylstm_%A_%a.err

set -eo pipefail
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$REPO_ROOT/train/pretrain_source.py" ]]; then
  echo "REPO_ROOT=$REPO_ROOT is not the repository root." >&2
  exit 2
fi
cd "$REPO_ROOT"
mkdir -p logs

source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
FOLD="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG="${CONFIG:?set CONFIG to a configs/phase1_dailylstm_aligned_*.yaml}"

export WANDB_DIR="${WANDB_DIR:-/ibex/user/kongw0a/global_mtslstm_wandb}"
mkdir -p "$WANDB_DIR"
[[ -n "${WANDB_API_KEY:-}" ]] || export WANDB_MODE=disabled

# This arm must never start from weights. --scratch is what makes it the baseline; a
# forgotten PRETRAIN_ROOT in the environment would silently turn it into a second M1.
if [[ -n "${PRETRAIN_ROOT:-}" ]]; then
  echo "PRETRAIN_ROOT is set, but this arm trains from scratch. Unset it." >&2
  exit 2
fi

OUT_ROOT_CFG=$("$PYTHON_BIN" -c "
from common.config import load_config, resolve
print(resolve(load_config('$CONFIG', None).output_root))")
if [[ -f "$OUT_ROOT_CFG/fold$FOLD/transfer/DONE" && "${FORCE:-0}" != "1" ]]; then
  echo "fold $FOLD already complete under $OUT_ROOT_CFG -- refusing to overwrite." >&2
  exit 0
fi

echo "fold=$FOLD config=$CONFIG out=$OUT_ROOT_CFG scratch=yes node=$(hostname)"
"$PYTHON_BIN" -u -m train.transfer_target --config "$CONFIG" \
  --fold "$FOLD" --scratch --score-branch daily --peek-hourly \
  --set "wandb.group=dailylstm_baseline"
