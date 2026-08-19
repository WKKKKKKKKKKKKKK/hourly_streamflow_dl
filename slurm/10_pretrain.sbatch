#!/bin/bash
# Phase I Step 1 -- source-domain pretraining, one array task per fold.
#   sbatch slurm/10_pretrain.sbatch                 # all 5 folds
#   sbatch --array=0 slurm/10_pretrain.sbatch       # just fold 0
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
# 4 hours is not a guess -- it is the gpu4 partition boundary, and that boundary
# dominates queue latency far more than fairshare does. Partitions on this cluster are
# tiered by MaxTime: gpu4 (<=4 h) has 95 nodes and the most free CPUs on the machine,
# while gpu24 / gpu72 / gpu have 46 / 31 / 58. A job asking for even 5 h is excluded
# from gpu4 and must compete in pools less than half the size. Measured: twelve 2-3 h
# jobs submitted 2026-08-18 all started within 0-31 min, while 5 h and 9 h jobs from
# the same account, same fairshare, same day waited over 24 hours.
#
# One pass of 30 epochs takes ~9 h, so a full run does NOT fit in one job. Chain
# several of these instead, each resuming from the last checkpoint:
#   H=$(sbatch --parsable slurm/10_pretrain.sbatch)
#   for i in 1 2; do H=$(sbatch --parsable --dependency=afterany:$H slurm/10_pretrain.sbatch); done
# afterany, not afterok: a chunk hitting the wall exits non-zero by design, and the
# next chunk is exactly what should handle that. --resume picks up the checkpoint, and
# the DONE guard makes surplus chunks exit 0 in seconds once the work is finished, so
# over-provisioning the chain is nearly free. Point any downstream afterok at the LAST
# chunk.
# GPU type: accept either. v100 is the most contended type on this cluster (85%
# utilised vs a100's 61%), and pinning it halved the pool of eligible nodes (23 vs 46
# for a 96G/8-CPU job) and narrowed the partition list. An untyped gres also opens
# gpu1 and gpu4. Verified on a100 (sm_80, torch 2.7.1+cu118): the model builds and
# trains there.
#SBATCH --constraint="v100|a100"
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH -J gmts_pre
#SBATCH -o logs/pretrain_%A_%a.out
#SBATCH -e logs/pretrain_%A_%a.err

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


# These two scripts had NEITHER a --config flag NOR an overwrite guard: CONFIG=... was
# silently ignored and the run wrote into whatever output_root the default config named.
# A v2 submission would therefore have overwritten outputs/runA_regwin24, the source of
# already-reported results. Same fix as the runB scripts.
CONFIG="${CONFIG:-configs/phase1.yaml}"
# Arrays do not survive --export=ALL; use a whitespace-separated string.
read -r -a EXTRA_SET_ARR <<< "${EXTRA_SET:-}"
OUT_ROOT_CFG=$("$PYTHON_BIN" -c "
from common.config import load_config, resolve
print(resolve(load_config('$CONFIG', None).output_root))")
# Guard on DONE, not on best_model.pth: the latter is rewritten every time the metric
# improves, so it exists mid-run too. Keying the guard on it made a legitimate --resume
# look like an overwrite, forcing FORCE=1 -- which would also switch off the protection
# for folds that really are finished.
if [[ -f "$OUT_ROOT_CFG/fold$FOLD/pretrain/DONE" && "${FORCE:-0}" != "1" ]]; then
  echo "fold $FOLD already complete under $OUT_ROOT_CFG -- refusing to overwrite." >&2
  echo "Point the variant at its own output_root, or set FORCE=1 to redo it." >&2
  exit 3
fi
echo "config=$CONFIG out=$OUT_ROOT_CFG fold=$FOLD"

"$PYTHON_BIN" -u -m train.pretrain_source \
  --config "$CONFIG" \
  --fold "$FOLD" \
  --resume \
  --set "wandb.group=phase1_pretrain" ${EXTRA_SET_ARR[@]+"${EXTRA_SET_ARR[@]}"}
