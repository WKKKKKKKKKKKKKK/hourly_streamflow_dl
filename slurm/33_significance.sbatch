#!/bin/bash
# Per-station paired Wilcoxon + Benjamini-Hochberg (PLAN.md 5).
#   CONFIG=configs/phase1_runB.yaml sbatch slurm/33_significance.sbatch
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --mem=128G --time=03:00:00
# GPU type: accept either. v100 is the most contended type on this cluster (85%
# utilised vs a100's 61%), and pinning it halved the pool of eligible nodes (23 vs 46
# for a 96G/8-CPU job) and narrowed the partition list. An untyped gres also opens
# gpu1 and gpu4. Verified on a100 (sm_80, torch 2.7.1+cu118): the model builds and
# trains there.
#SBATCH --constraint="v100|a100" --gres=gpu:1
#SBATCH -J gmts_signif -o logs/signif_%j.out -e logs/signif_%j.err
set -eo pipefail
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$REPO_ROOT/configs/phase1.yaml" ]] || { echo "not the repo: $REPO_ROOT" >&2; exit 2; }
cd "$REPO_ROOT"; mkdir -p logs
source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
CONFIG="${CONFIG:-configs/phase1_runB.yaml}"

# Refuse to clobber a finished run. This script had NO guard and its default config is
# v1's (outputs/runB_truedaily), so a submission whose CONFIG failed to propagate would
# silently overwrite the v1 numbers that RESULTS_phase1.md quotes. Key on
# significance_summary.json, which the script writes LAST, so an interrupted run is
# still re-runnable. FORCE=1 to override deliberately.
OUT_ROOT_CFG=$("$PYTHON_BIN" -c "
from common.config import load_config, resolve
print(resolve(load_config('$CONFIG', None).output_root))")
if [[ -f "$OUT_ROOT_CFG/significance/significance_summary.json" && "${FORCE:-0}" != "1" ]]; then
  echo "REFUSING: $OUT_ROOT_CFG/significance/significance_summary.json exists (FORCE=1 to override)"
  exit 0
fi
echo "config=$CONFIG out=$OUT_ROOT_CFG/significance"
PRE=(); [[ -n "${PRETRAIN_ROOT:-}" ]] && PRE=(--pretrain-dir "$PRETRAIN_ROOT")
INDEX=(); [[ -n "${INDEX:-}" ]] && INDEX=(--set "data.eval_sample_index=$INDEX")
echo "config=$CONFIG node=$(hostname)"
"$PYTHON_BIN" -u -m scripts.significance --config "$CONFIG" \
  --folds "${FOLDS:-0,1,2,3,4}" "${PRE[@]}" ${INDEX[@]+"${INDEX[@]}"}
