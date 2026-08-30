#!/bin/bash
# Fold-1 hyperparameter search (PLAN.md 2). One variant per submission.
#   CONFIG=configs/search/dropout00.yaml sbatch slurm/50_search.sbatch
# 4 hours, not 36. These folds take 4.6-7 h, and a 36 h request confined them to the
# gpu/gpu72 partitions where the scheduler predicted a 3-5 DAY wait, while gpu4 sat
# with 79 free nodes. Chunking with --resume and chaining on afterany is the pattern
# 10_pretrain_runB.sbatch already uses for exactly this reason: a wall-clock kill exits
# non-zero BY DESIGN, so the chain must be afterany, and the DONE guard makes surplus
# chunks exit 0 in seconds once the fold is finished.
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --mem=96G --time=04:00:00
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
# Guard on DONE, not on best_model.pth: the latter is rewritten every time the metric
# improves, so it exists mid-run too, and a chunked --resume would see it and exit as
# though the fold were finished. 10_pretrain_runB.sbatch already learned this; this
# script had kept the older, wrong guard because it had only ever run unchunked.
if [[ -f "$OUT/fold$FOLD/pretrain/DONE" && "${FORCE:-0}" != "1" ]]; then
  # exit 0: "already done" is success for an idempotent job, and a non-zero code here
  # breaks downstream --dependency=afterok. The two guards in this repo that DO exit
  # non-zero (41_africa, 40_basin_average) refuse for a different reason -- a required
  # input is missing -- which is a real failure.
  echo "$NAME fold $FOLD already done under $OUT -- refusing to overwrite." >&2; exit 0
fi
echo "variant=$NAME fold=$FOLD out=$OUT node=$(hostname)"

"$PYTHON_BIN" -u -m train.pretrain_source --config "$CONFIG" --fold "$FOLD" --resume \
  --set "wandb.group=phase1_search" "wandb.mode=${WANDB_MODE:-online}"
"$PYTHON_BIN" -u -m train.transfer_target --config "$CONFIG" --fold "$FOLD" --peek-hourly \
  --set "wandb.group=phase1_search"
