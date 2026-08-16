#!/bin/bash
# KGE decomposition (r / alpha / beta) for M0 vs M1 -- evaluation only, no training.
# One array task per fold, then merge:
#   D=$(sbatch --parsable slurm/30_diagnose.sbatch)
#   sbatch --dependency=afterany:$D slurm/30_diagnose.sbatch   # with MERGE=1
# Override the run with CONFIG=/RUN_DIR=, e.g.
#   CONFIG=configs/phase1_runB.yaml sbatch slurm/30_diagnose.sbatch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00      # two forward passes over the same eval set as the transfer step
#SBATCH --constraint=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-4
#SBATCH -J gmts_diag
#SBATCH -o logs/diagnose_%A_%a.out
#SBATCH -e logs/diagnose_%A_%a.err

set -eo pipefail
# SLURM copies this script to a node-local spool directory, so BASH_SOURCE does
# NOT point at the repo. Use the submit directory, or GMTS_REPO if you submit from
# elsewhere, and verify it really is the repo before touching anything.
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$REPO_ROOT/configs/phase1.yaml" ]]; then
  echo "REPO_ROOT=$REPO_ROOT is not the global_mtslstm repo (no configs/phase1.yaml)." >&2
  echo "Submit from the repo root, or export GMTS_REPO=/path/to/global_mtslstm." >&2
  exit 2
fi
cd "$REPO_ROOT"
mkdir -p logs

source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

CONFIG="${CONFIG:-configs/phase1.yaml}"
DOMAIN="${DOMAIN:-target}"
FOLD="${SLURM_ARRAY_TASK_ID:-0}"
RUN_DIR_ARG=()
[[ -n "${RUN_DIR:-}" ]] && RUN_DIR_ARG=(--run-dir "$RUN_DIR")
# A variant that reused another run's pretrained weights has no pretrain/ of its
# own, so M0 must be read from the run it started FROM.
[[ -n "${PRETRAIN_ROOT:-}" ]] && RUN_DIR_ARG+=(--pretrain-dir "$PRETRAIN_ROOT")
# INDEX swaps the cache path's sample index for scoring only. The stride-24
# training index is 23:00-only, so leaving it in place reports an "hourly" KGE
# over one hour of the day; INDEX=samples_evalhours.npz covers all 24.
SET_ARG=()
[[ -n "${INDEX:-}" ]] && SET_ARG=(--set "data.eval_sample_index=$INDEX")

echo "config=$CONFIG domain=$DOMAIN fold=$FOLD index=${INDEX:-<training>} merge=${MERGE:-0} node=$(hostname)"

if [[ "${MERGE:-0}" == "1" ]]; then
  # Aggregation only -- reads the per-fold CSVs, needs no GPU and no data.
  "$PYTHON_BIN" -u -m scripts.diagnose_kge \
    --config "$CONFIG" --domain "$DOMAIN" --merge "${RUN_DIR_ARG[@]}" \
    ${OUT_ROOT:+--out-dir "$OUT_ROOT"}
  exit $?
fi

OUT_ROOT="${OUT_ROOT:-}"
if [[ -z "$OUT_ROOT" ]]; then
  # Mirror the script's default (output_root/diagnostics) but one level per fold,
  # so the five array tasks cannot overwrite each other's CSVs.
  OUT_ROOT=$("$PYTHON_BIN" -c "
from pathlib import Path
from common.config import load_config, resolve
cfg = load_config('$CONFIG', None)
print(Path(resolve(cfg.output_root)) / 'diagnostics')")
fi

"$PYTHON_BIN" -u -m scripts.diagnose_kge \
  --config "$CONFIG" --domain "$DOMAIN" --folds "$FOLD" \
  --out-dir "$OUT_ROOT/fold$FOLD" "${RUN_DIR_ARG[@]}" ${SET_ARG[@]+"${SET_ARG[@]}"}
