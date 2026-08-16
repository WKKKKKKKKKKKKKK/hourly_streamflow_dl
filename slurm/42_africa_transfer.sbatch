#!/bin/bash
# Phase I protocol run in situ on African basins: fine-tune on their own daily
# observations, then score daily skill on their held-out period.
#   CONFIG=configs/phase1_runB.yaml FOLD=0 sbatch slurm/42_africa_transfer.sbatch
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --mem=96G --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH -J gmts_afxfer -o logs/afxfer_%j.out -e logs/afxfer_%j.err
set -eo pipefail
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$REPO_ROOT/configs/phase1.yaml" ]] || { echo "not the repo: $REPO_ROOT" >&2; exit 2; }
cd "$REPO_ROOT"; mkdir -p logs
source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
export WANDB_DIR="${WANDB_DIR:-/ibex/user/kongw0a/global_mtslstm_wandb}"; mkdir -p "$WANDB_DIR"
CONFIG="${CONFIG:-configs/phase1_runB.yaml}"
FOLD="${FOLD:-0}"
PRE=(); [[ -n "${PRETRAINED:-}" ]] && PRE=(--pretrained "$PRETRAINED")
OUT=(); [[ -n "${OUT_DIR:-}" ]] && OUT=(--out-dir "$OUT_DIR")
echo "config=$CONFIG fold=$FOLD node=$(hostname)"
"$PYTHON_BIN" -u -m train.africa_transfer --config "$CONFIG" --fold "$FOLD" "${PRE[@]}" "${OUT[@]}"
