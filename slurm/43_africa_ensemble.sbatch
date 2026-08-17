#!/bin/bash
# Ensemble the five in-situ African models by averaging their daily predictions,
# matching how eval.africa reports (average predictions, then score once).
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --mem=96G --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH -J gmts_afens -o logs/afens_%j.out -e logs/afens_%j.err
set -eo pipefail
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$REPO_ROOT/configs/phase1.yaml" ]] || { echo "not the repo: $REPO_ROOT" >&2; exit 2; }
cd "$REPO_ROOT"; mkdir -p logs
source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
CONFIG="${CONFIG:-configs/phase1_runB.yaml}"
echo "config=$CONFIG node=$(hostname)"
"$PYTHON_BIN" -u -m scripts.africa_insitu_ensemble --config "$CONFIG"
