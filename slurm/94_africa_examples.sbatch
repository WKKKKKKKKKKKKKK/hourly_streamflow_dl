#!/bin/bash
# Daily series of every African catchment (5 members x 2 arms) and hourly ones of a few, for the African examples
# (scripts/analysis/africa_example_series.py). Inference only, all 294 basins x 5 members x 2 arms; too slow on a login node.
#   sbatch slurm/94_africa_examples.sbatch            # hourly kept for the M1 KGE percentile candidates
#   STATIONS=a,b,c sbatch slurm/94_africa_examples.sbatch
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --mem=96G --time=01:30:00
#SBATCH --gres=gpu:1 --constraint="v100|a100"
#SBATCH -J al_afex -o logs/afex_%j.out -e logs/afex_%j.err
set -eo pipefail
cd "${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
ARGS=(); [[ -n "${STATIONS:-}" ]] && ARGS=(--stations "$STATIONS")
"$PYTHON_BIN" -u -m scripts.analysis.africa_example_series "${ARGS[@]}"
