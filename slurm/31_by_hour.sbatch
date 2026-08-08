#!/bin/bash
# Hourly KGE broken down by hour of day, on the all-hours evaluation index.
# Needs samples_evalhours.npz (slurm/04_build_eval_index.sbatch) to exist.
#   CONFIG=configs/phase1_runB.yaml sbatch slurm/31_by_hour.sbatch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --constraint=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-4
#SBATCH -J gmts_byhour
#SBATCH -o logs/byhour_%A_%a.out
#SBATCH -e logs/byhour_%A_%a.err

set -eo pipefail
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$REPO_ROOT/configs/phase1.yaml" ]]; then
  echo "REPO_ROOT=$REPO_ROOT is not the global_mtslstm repo (no configs/phase1.yaml)." >&2
  exit 2
fi
cd "$REPO_ROOT"
mkdir -p logs

source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

CONFIG="${CONFIG:-configs/phase1_runB.yaml}"
DOMAIN="${DOMAIN:-target}"
INDEX="${INDEX:-samples_evalhours.npz}"
FOLD="${SLURM_ARRAY_TASK_ID:-0}"

echo "config=$CONFIG domain=$DOMAIN index=$INDEX fold=$FOLD node=$(hostname)"

"$PYTHON_BIN" -u -m scripts.kge_by_hour \
  --config "$CONFIG" --fold "$FOLD" --domain "$DOMAIN" \
  --set "data.eval_sample_index=$INDEX"
