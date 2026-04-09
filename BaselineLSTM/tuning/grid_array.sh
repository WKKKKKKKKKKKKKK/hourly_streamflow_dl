#!/bin/bash
#SBATCH -N 1
#SBATCH --constraint=v100
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -J BaselineLSTM_grid
#SBATCH -o logs/grid_%A_%a.out
#SBATCH -e logs/grid_%A_%a.err

# Submit with:
#   sbatch --array=1-$(cat grid_count.txt)%<PARALLEL> grid_array.sh
# Resubmit the same array index to resume the same run/output-dir.

set -eo pipefail
EXP_DIR="/home/kongw0a/MTS_LSTM/experiment_withcursor/BaselineLSTM"
cd "${EXP_DIR}"
mkdir -p logs runs

# Optional: uncomment to force V100 only

if [[ ! -f grid_runs.jsonl ]]; then
  echo "grid_runs.jsonl not found; generating..."
  python make_grid.py
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Use sbatch --array=..." >&2
  exit 2
fi

python run_grid.py --idx "${SLURM_ARRAY_TASK_ID}"

