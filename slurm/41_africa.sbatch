#!/bin/bash
# Phase I Steps 4-5 -- African daily validation. Out of sample for every fold.
#   CONFIG=configs/phase1_runB.yaml sbatch slurm/41_africa.sbatch
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --mem=96G --time=06:00:00
#SBATCH --constraint=v100 --gres=gpu:v100:1
#SBATCH -J gmts_africa -o logs/africa_%j.out -e logs/africa_%j.err
set -eo pipefail
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$REPO_ROOT/configs/phase1.yaml" ]] || { echo "not the repo: $REPO_ROOT" >&2; exit 2; }
cd "$REPO_ROOT"; mkdir -p logs
source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

CONFIG="${CONFIG:-configs/phase1_runB.yaml}"
# The rescaled file, not the raw ERA5-Land one: raw pev is 2.29x the Penman PET the
# model trained on, which alone puts the input 2.5 sigma out of distribution.
FORCING="${FORCING:-/ibex/user/kongw0a/era5_land_africa_forcing/era5_land_africa_hourly_forcing_penman.nc}"
OUT="${OUT:-outputs/africa}"
KIND="${KIND:-transfer}"
[[ -f "$FORCING" ]] || { echo "forcing $FORCING missing -- run scripts.rescale_africa_forcing" >&2; exit 3; }

echo "config=$CONFIG kind=$KIND forcing=$(basename "$FORCING") out=$OUT node=$(hostname)"
"$PYTHON_BIN" -u -m eval.africa --config "$CONFIG" --forcing "$FORCING" \
  --checkpoint-kind "$KIND" --out-dir "$OUT"
