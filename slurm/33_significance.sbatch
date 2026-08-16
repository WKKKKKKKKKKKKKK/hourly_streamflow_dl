#!/bin/bash
# Per-station paired Wilcoxon + Benjamini-Hochberg (PLAN.md 5).
#   CONFIG=configs/phase1_runB.yaml sbatch slurm/33_significance.sbatch
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --mem=128G --time=03:00:00
#SBATCH --constraint=v100 --gres=gpu:v100:1
#SBATCH -J gmts_signif -o logs/signif_%j.out -e logs/signif_%j.err
set -eo pipefail
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$REPO_ROOT/configs/phase1.yaml" ]] || { echo "not the repo: $REPO_ROOT" >&2; exit 2; }
cd "$REPO_ROOT"; mkdir -p logs
source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
CONFIG="${CONFIG:-configs/phase1_runB.yaml}"
PRE=(); [[ -n "${PRETRAIN_ROOT:-}" ]] && PRE=(--pretrain-dir "$PRETRAIN_ROOT")
INDEX=(); [[ -n "${INDEX:-}" ]] && INDEX=(--set "data.eval_sample_index=$INDEX")
echo "config=$CONFIG node=$(hostname)"
"$PYTHON_BIN" -u -m scripts.significance --config "$CONFIG" \
  --folds "${FOLDS:-0,1,2,3,4}" "${PRE[@]}" ${INDEX[@]+"${INDEX[@]}"}
