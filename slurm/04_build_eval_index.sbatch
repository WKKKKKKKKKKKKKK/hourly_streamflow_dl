#!/bin/bash
# All-hours evaluation index for the cache path (scripts/build_eval_index.py).
# CPU only -- it scans forcing.f32 and writes samples_evalhours.npz.
#   sbatch slurm/04_build_eval_index.sbatch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH -J gmts_evalidx
#SBATCH -o logs/evalindex_%j.out
#SBATCH -e logs/evalindex_%j.err

set -eo pipefail
# SLURM copies this script to a node-local spool directory, so BASH_SOURCE does
# NOT point at the repo. Use the submit directory, or GMTS_REPO if you submit from
# elsewhere, and verify it really is the repo before touching anything.
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$REPO_ROOT/configs/phase1.yaml" ]]; then
  echo "REPO_ROOT=$REPO_ROOT is not the global_mtslstm repo (no configs/phase1.yaml)." >&2
  exit 2
fi
cd "$REPO_ROOT"
mkdir -p logs

source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

echo "per_station=${PER_STATION:-6144} node=$(hostname)"
"$PYTHON_BIN" -u -m scripts.build_eval_index --per-station "${PER_STATION:-6144}"
