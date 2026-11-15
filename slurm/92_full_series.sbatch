#!/bin/bash
# Full-series hourly evaluation and flood-peak metrics (scripts/full_series_eval.py).
#   CONFIG=configs/aligned/phase1_runB_v2.yaml FOLDS=0 sbatch slurm/92_full_series.sbatch
# One fold per job. Never pass a list through --export (FOLDS=0,1,2): sbatch splits --export
# at commas, so the job silently receives FOLDS=0 and evaluates fold 0 only.
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --mem=110G --time=04:00:00
#SBATCH --constraint="v100|a100" --gres=gpu:1
#SBATCH -J al_fullser -o logs/fullser_%j.out -e logs/fullser_%j.err
set -eo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO_ROOT"
[[ -f train/pretrain_source.py ]] || { echo "not the repo" >&2; exit 2; }
export WANDB_MODE=disabled
CONFIG="${CONFIG:?set CONFIG}"
EXTRA=(); [[ -n "${FOLDS:-}" ]] && EXTRA+=(--folds "$FOLDS")
[[ -n "${MAX_STATIONS:-}" ]] && EXTRA+=(--max-stations "$MAX_STATIONS")
[[ -n "${OUT_DIR:-}" ]] && EXTRA+=(--out-dir "$OUT_DIR")
echo "config=$CONFIG ${EXTRA[*]} node=$(hostname)"
"$(command -v python3)" -u -m scripts.analysis.full_series_eval --config "$CONFIG" ${EXTRA[@]+"${EXTRA[@]}"}
