#!/bin/bash
# Phase I protocol run in situ on African basins: fine-tune on their own daily
# observations, then score daily skill on their held-out period.
#   CONFIG=configs/aligned/phase1_runB_v2.yaml FOLD=0 sbatch slurm/42_africa_transfer.sbatch
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --mem=96G --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH -J gmts_afxfer -o logs/afxfer_%j.out -e logs/afxfer_%j.err
set -eo pipefail
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$REPO_ROOT/train/pretrain_source.py" ]] || { echo "not the repo: $REPO_ROOT" >&2; exit 2; }
cd "$REPO_ROOT"; mkdir -p logs
source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
export WANDB_DIR="${WANDB_DIR:-/ibex/user/kongw0a/global_mtslstm_wandb}"; mkdir -p "$WANDB_DIR"
CONFIG="${CONFIG:?set CONFIG=configs/...yaml}"
FOLD="${FOLD:-0}"
PRE=(); [[ -n "${PRETRAINED:-}" ]] && PRE=(--pretrained "$PRETRAINED")
OUT=(); [[ -n "${OUT_DIR:-}" ]] && OUT=(--out-dir "$OUT_DIR")
# Baseline switches. SCRATCH=1 skips the pretrained checkpoint so the run measures the
# same architecture trained on African daily observations alone; SCORE_BRANCH=daily reads
# the daily LSTM's own head. EPOCHS/PATIENCE are overridable because a from-scratch run
# needs a far larger budget than a fine-tune, and an under-trained baseline would make the
# comparison meaningless in the direction that flatters us.
EXTRA=()
[[ "${SCRATCH:-0}" == "1" ]] && EXTRA+=(--scratch)
[[ -n "${SCORE_BRANCH:-}" ]] && EXTRA+=(--score-branch "$SCORE_BRANCH")
[[ -n "${SELECT_FRAC:-}" ]] && EXTRA+=(--select-frac "$SELECT_FRAC")
[[ -n "${EPOCHS:-}" ]] && EXTRA+=(--epochs "$EPOCHS")
[[ -n "${PATIENCE:-}" ]] && EXTRA+=(--patience "$PATIENCE")
echo "config=$CONFIG fold=$FOLD extra=${EXTRA[*]:-<none>} node=$(hostname)"
nvidia-smi -L >/dev/null 2>&1 || { echo "no CUDA device visible -- refusing to run" >&2; exit 2; }
"$PYTHON_BIN" -u -m train.africa_transfer --config "$CONFIG" --fold "$FOLD" \
  "${PRE[@]}" "${OUT[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
