#!/bin/bash
# One-off: cache 6sources.nc as a memmap plus the daily means and sample index.
# Not on a login node: a first attempt there was killed silently around station
# 5,600 while writing the 55 GiB memmap. Resumes from wherever it stopped.
#   sbatch slurm/01_build_cache.sbatch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -J gmts_cache
#SBATCH -o logs/cache_%j.out
#SBATCH -e logs/cache_%j.err

set -eo pipefail
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$REPO_ROOT/train/pretrain_source.py" ]]; then
  echo "REPO_ROOT=$REPO_ROOT is not the repository root" >&2; exit 2
fi
cd "$REPO_ROOT"; mkdir -p logs
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
echo "node=$(hostname) python=$PYTHON_BIN"
df -h /ibex/user | tail -1

# Full build (writes forcing.f32, ~55 GiB). SKIP_MEMMAP=1 rebuilds only the sample index.
SKIP=(); [[ "${SKIP_MEMMAP:-0}" == 1 ]] && SKIP=(--skip-memmap)
"$PYTHON_BIN" -u -m scripts.prep.build_hourly_cache --stride 24 ${SKIP[@]+"${SKIP[@]}"}
