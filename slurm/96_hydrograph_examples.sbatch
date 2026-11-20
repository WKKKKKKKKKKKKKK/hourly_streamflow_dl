#!/bin/bash
# Hourly series of the five hydrograph example gauges, M0 and M1 over the whole test period
# (scripts/figures/fig_hydrograph_examples.py --predict-only). Inference only; slow on a login node.
#   DATA_ROOT=<dir with scalers.json and config.json> sbatch slurm/96_hydrograph_examples.sbatch
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --mem=96G --time=01:00:00
#SBATCH --gres=gpu:1 --constraint="v100|a100"
#SBATCH -J al_hydex -o logs/hydex_%j.out -e logs/hydex_%j.err
set -eo pipefail
cd "${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
export DATA_ROOT="${DATA_ROOT:-/ibex/user/kongw0a/dec_track/data/june_check}"
"$PYTHON_BIN" -u -m scripts.figures.fig_hydrograph_examples --predict-only
