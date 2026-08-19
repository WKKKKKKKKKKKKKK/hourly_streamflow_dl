#!/bin/bash
# Fold-1 hyperparameter search (PLAN.md 2). One variant per submission.
#   CONFIG=configs/search/dropout00.yaml sbatch slurm/50_search.sbatch
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --mem=96G --time=36:00:00
# GPU type: accept either. v100 is the most contended type on this cluster (85%
# utilised vs a100's 61%), and pinning it halved the pool of eligible nodes (23 vs 46
# for a 96G/8-CPU job) and narrowed the partition list. An untyped gres also opens
# gpu1 and gpu4. Verified on a100 (sm_80, torch 2.7.1+cu118): the model builds and
# trains there.
#SBATCH --constraint="v100|a100" --gres=gpu:1
#SBATCH -J gmts_search -o logs/search_%j.out -e logs/search_%j.err
set -eo pipefail
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$REPO_ROOT/configs/phase1.yaml" ]] || { echo "not the repo: $REPO_ROOT" >&2; exit 2; }
cd "$REPO_ROOT"; mkdir -p logs
source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
export WANDB_DIR="${WANDB_DIR:-/ibex/user/kongw0a/global_mtslstm_wandb}"; mkdir -p "$WANDB_DIR"

CONFIG="${CONFIG:?set CONFIG=configs/search/<name>.yaml}"
FOLD="${FOLD:-1}"
NAME=$(basename "$CONFIG" .yaml)
OUT=$("$PYTHON_BIN" -c "
from common.config import load_config, resolve
print(resolve(load_config('$CONFIG', None).output_root))")
if [[ -f "$OUT/fold$FOLD/pretrain/best_model.pth" && "${FORCE:-0}" != "1" ]]; then
  echo "$NAME fold $FOLD already done under $OUT -- refusing to overwrite." >&2; exit 3
fi
echo "variant=$NAME fold=$FOLD out=$OUT node=$(hostname)"

"$PYTHON_BIN" -u -m train.pretrain_source --config "$CONFIG" --fold "$FOLD" --resume \
  --set "wandb.group=phase1_search" "wandb.mode=${WANDB_MODE:-online}"
"$PYTHON_BIN" -u -m train.transfer_target --config "$CONFIG" --fold "$FOLD" --peek-hourly \
  --set "wandb.group=phase1_search"
