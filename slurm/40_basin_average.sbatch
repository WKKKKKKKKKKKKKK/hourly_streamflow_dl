#!/bin/bash
# Turn the downloaded ERA5-Land grids into per-basin series for the African basins.
# PRESET=forcing -> hourly pet/pcp/temp that drive the model (Step 4 input)
# PRESET=runoff  -> daily ERA5-Land runoff, the physical baseline (Step 5)
#   PRESET=forcing sbatch slurm/40_basin_average.sbatch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH -J gmts_basinavg
#SBATCH -o logs/basinavg_%j.out
#SBATCH -e logs/basinavg_%j.err

set -eo pipefail
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$REPO_ROOT/configs/phase1.yaml" ]]; then
  echo "REPO_ROOT=$REPO_ROOT is not the global_mtslstm repo." >&2
  exit 2
fi
cd "$REPO_ROOT"
mkdir -p logs

source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

PRESET="${PRESET:-forcing}"
case "$PRESET" in
  forcing) MODULE=scripts.basin_average_era5_forcing; DIR=/ibex/user/kongw0a/era5_land_africa_forcing ;;
  runoff)  MODULE=scripts.basin_average_era5_land;    DIR=/ibex/user/kongw0a/era5_land_africa ;;
  *) echo "PRESET must be forcing|runoff, got $PRESET" >&2; exit 2 ;;
esac

N=$(ls "$DIR"/*.nc 2>/dev/null | wc -l)
echo "preset=$PRESET module=$MODULE dir=$DIR files=$N node=$(hostname)"
if [[ "$N" -eq 0 ]]; then echo "no .nc files in $DIR" >&2; exit 3; fi

"$PYTHON_BIN" -u -m "$MODULE" --era5-dir "$DIR"
