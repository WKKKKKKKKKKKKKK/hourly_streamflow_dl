#!/bin/bash
# Save the HOURLY African predictions for the three catchments figure 7 shows.
# The daily ensemble discards them; nothing else in the project keeps an hourly
# African series, so there is currently no hourly African figure that can be drawn.
# Small by construction: 3 basins, not 294. The 1h walltime routes it to gpu4
# (<=4h, 95 nodes) automatically -- the partition cannot be named directly.
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=01:00:00
#SBATCH --gres=gpu:1 --constraint="v100|a100"
#SBATCH -J gmts_afhr -o logs/afhr_%j.out -e logs/afhr_%j.err
set -eo pipefail
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$REPO_ROOT/configs/phase1.yaml" ]] || { echo "not the repo: $REPO_ROOT" >&2; exit 2; }
cd "$REPO_ROOT"; mkdir -p logs
source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
# v2 is the primary configuration: lookback_hourly 336 and initial_forget_bias 3.
# Running this under the v1 config would silently feed the v2 checkpoints a 72-step
# hourly branch -- a different input, not a different setting.
CONFIG="${CONFIG:-configs/phase1_runB_v2.yaml}"
echo "config=$CONFIG node=$(hostname)"
# ALL_BASINS=1 adds the within-day statistics over every African basin (~2 min more on
# one GPU). Without it the pass covers only the three catchments the figure draws, which
# is enough for the figure but cannot support a claim about the model.
ALL=""; [[ -n "${ALL_BASINS:-}" ]] && ALL="--all-basins"
"$PYTHON_BIN" -u -m scripts.africa_hourly_series \
  --config "$CONFIG" $ALL \
  --insitu-glob outputs/v2_africa_insitu_fold \
  --pretrain-root outputs/v2_runB \
  --per-basin outputs/v2_africa_insitu_summary/ensemble_per_basin_M1.csv \
  --out-dir outputs/v2_africa_hourly
