#!/bin/bash
# Phase I Steps 2-3 -- daily-only target transfer + source-degradation check.
# Needs the matching fold's pretrain output to exist.
#   sbatch --dependency=afterok:<pretrain_jobid> slurm/20_transfer.sbatch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00      # measured ~3 h/fold, dominated by the final eval passes
# GPU type: accept either. v100 is the most contended type on this cluster (85%
# utilised vs a100's 61%), and pinning it halved the pool of eligible nodes (23 vs 46
# for a 96G/8-CPU job) and narrowed the partition list. An untyped gres also opens
# gpu1 and gpu4. Verified on a100 (sm_80, torch 2.7.1+cu118): the model builds and
# trains there.
#SBATCH --constraint="v100|a100"
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH -J gmtsB_xfer
#SBATCH -o logs/transferB_%A_%a.out
#SBATCH -e logs/transferB_%A_%a.err

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

CONFIG="${CONFIG:-configs/phase1_runB.yaml}"
WANDB_GROUP="${WANDB_GROUP:-phase1B_transfer}"
# A variant run (e.g. source replay) writes to its own output_root but must start
# from the SAME pretrained weights, otherwise the comparison measures two
# pretrainings rather than the one thing that changed. PRETRAIN_ROOT points at the
# run whose fold{N}/pretrain/best_model.pth to load.
PRETRAINED_ARG=()
if [[ -n "${PRETRAIN_ROOT:-}" ]]; then
  CKPT="$PRETRAIN_ROOT/fold$FOLD/pretrain/best_model.pth"
  if [[ ! -f "$CKPT" ]]; then
    echo "PRETRAIN_ROOT set but $CKPT does not exist." >&2
    exit 2
  fi
  PRETRAINED_ARG=(--pretrained "$CKPT")
fi

# EXTRA_SET is a whitespace-separated STRING, not a bash array. Arrays do not
# survive --export=ALL (the environment holds strings only), so an array-valued
# EXTRA_SET arrives empty and the job silently runs the config's defaults --
# which, for a variant sharing one output_root, would overwrite finished results.
#   EXTRA_SET="transfer.source_replay_ratio=0.5" sbatch slurm/20_transfer_runB.sbatch
read -r -a EXTRA_SET_ARR <<< "${EXTRA_SET:-}"

# Refuse to clobber a finished fold. train.transfer_target writes DONE last, so its
# presence means this output_root already holds complete results for this fold.
OUT_ROOT_CFG=$("$PYTHON_BIN" -c "
from common.config import load_config, resolve
print(resolve(load_config('$CONFIG', None).output_root))")
if [[ -f "$OUT_ROOT_CFG/fold$FOLD/transfer/DONE" && "${FORCE:-0}" != "1" ]]; then
  echo "fold $FOLD already complete under $OUT_ROOT_CFG -- refusing to overwrite." >&2
  echo "Point the variant at its own output_root, or set FORCE=1 to redo it." >&2
  # exit 0, not 3. "Already finished" is success for an idempotent job, and a non-zero
  # code here breaks any downstream --dependency=afterok. That is not hypothetical: the
  # v3 chunk chain was built on the claim that surplus chunks "exit 0 in seconds", and
  # when two run A continuation chunks hit this guard they exited 3, failed, and would
  # have left four transfer jobs waiting on an afterok that could never be satisfied.
  # The refusal is still visible -- it is on stderr and in the log; the exit code is the
  # wrong channel for it.
  exit 0
fi

echo "fold=$FOLD config=$CONFIG out=$OUT_ROOT_CFG pretrained=${PRETRAIN_ROOT:-<config default>}"
echo "extra_set=${EXTRA_SET:-<none>} node=$(hostname)"

"$PYTHON_BIN" -u -m train.transfer_target --config "$CONFIG" \
  --fold "$FOLD" \
  --peek-hourly \
  "${PRETRAINED_ARG[@]}" \
  --set "wandb.group=$WANDB_GROUP" ${EXTRA_SET_ARR[@]+"${EXTRA_SET_ARR[@]}"}
