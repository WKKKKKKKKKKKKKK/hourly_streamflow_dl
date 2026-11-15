#!/bin/bash
# Daily-observation correction of M0, the post-processing baseline for M1 (scripts/correction_baseline.py).
#   CONFIG=configs/aligned/phase1_runB_blocked_v2.yaml FOLDS=0 sbatch slurm/95_correction_baseline.sbatch
# One fold per job (sbatch --export splits a list at commas). Inference only.
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --mem=96G --time=03:00:00
#SBATCH --gres=gpu:1 --constraint="v100|a100"
#SBATCH -J al_corr -o logs/corr_%j.out -e logs/corr_%j.err
set -eo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO_ROOT"
[[ -f train/pretrain_source.py ]] || { echo "not the repo" >&2; exit 2; }
export WANDB_MODE=disabled
CONFIG="${CONFIG:?set CONFIG}"
"$(command -v python3)" -u -m scripts.analysis.correction_baseline --config "$CONFIG" --folds "${FOLDS:?set FOLDS}"
