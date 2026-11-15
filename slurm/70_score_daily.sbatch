#!/bin/bash
# Daily-scale scoring of an existing transfer checkpoint on its target fold.
# Evaluation only: no training, no weights written, nothing existing overwritten.
#   CONFIG=configs/aligned/phase1_runB_v2.yaml sbatch slurm/70_score_daily.sbatch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --constraint="v100|a100"
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH -J gmts_scoreD
#SBATCH -o logs/scoreD_%A_%a.out
#SBATCH -e logs/scoreD_%A_%a.err

set -eo pipefail
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$REPO_ROOT/train/pretrain_source.py" ]]; then
  echo "REPO_ROOT=$REPO_ROOT is not the repository root." >&2
  exit 2
fi
cd "$REPO_ROOT"
mkdir -p logs

source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
FOLD="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG="${CONFIG:?set CONFIG to the config of the design being scored}"
TAG="${TAG:-M1}"
export WANDB_MODE=disabled

CKPT_ARG=()
[[ -n "${CHECKPOINT:-}" ]] && CKPT_ARG=(--checkpoint "$CHECKPOINT")
# PRETRAIN_ROOT scores the PRETRAINED weights instead of the adapted ones, which is how
# M0 gets a daily score: the fold number has to be interpolated inside the job, so it
# cannot be passed through CHECKPOINT from the submitting shell.
if [[ -n "${PRETRAIN_ROOT:-}" ]]; then
  CKPT="$PRETRAIN_ROOT/fold$FOLD/pretrain/best_model.pth"
  [[ -f "$CKPT" ]] || { echo "PRETRAIN_ROOT set but $CKPT does not exist." >&2; exit 2; }
  CKPT_ARG=(--checkpoint "$CKPT")
fi

# Which head the daily metric reads. It MUST match how the checkpoint was trained:
# a model whose hourly branch never had a gradient has no meaningful 24-h aggregate,
# and scoring it that way measures noise. Default agg; set daily for the baseline arm.
SCORE_BRANCH="${SCORE_BRANCH:-agg}"
# A source pass scores ~7,200 gauges against ~1,800 for a target pass.
DOMAIN="${DOMAIN:-target}"

echo "fold=$FOLD config=$CONFIG tag=$TAG domain=$DOMAIN score_branch=$SCORE_BRANCH node=$(hostname)"
"$PYTHON_BIN" -u -m scripts.analysis.score_daily_target --config "$CONFIG" \
  --fold "$FOLD" --tag "$TAG" --domain "$DOMAIN" --score-branch "$SCORE_BRANCH" "${CKPT_ARG[@]}"
