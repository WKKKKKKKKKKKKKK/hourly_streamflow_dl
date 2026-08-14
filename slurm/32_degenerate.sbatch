#!/bin/bash
# Intra-day shape check -- the degenerate-solution diagnostic (PLAN.md 5.4).
#   CONFIG=configs/phase1_runB.yaml sbatch slurm/32_degenerate.sbatch
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --mem=128G --time=06:00:00
#SBATCH --constraint=v100 --gres=gpu:v100:1
#SBATCH -J gmts_degen -o logs/degen_%j.out -e logs/degen_%j.err
set -eo pipefail
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$REPO_ROOT/configs/phase1.yaml" ]] || { echo "not the repo: $REPO_ROOT" >&2; exit 2; }
cd "$REPO_ROOT"; mkdir -p logs
source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
CONFIG="${CONFIG:-configs/phase1_runB.yaml}"
PRE=(); [[ -n "${PRETRAIN_ROOT:-}" ]] && PRE=(--pretrain-dir "$PRETRAIN_ROOT")
echo "config=$CONFIG folds=${FOLDS:-0,1,2,3,4} node=$(hostname)"
"$PYTHON_BIN" -u -m scripts.degenerate_check --config "$CONFIG" \
  --folds "${FOLDS:-0,1,2,3,4}" "${PRE[@]}"
