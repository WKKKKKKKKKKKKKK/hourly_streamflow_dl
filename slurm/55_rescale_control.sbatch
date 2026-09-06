#!/bin/bash
# The control that decides how large a claim Phase I can make: is the daily-only gain
# anything more than a per-gauge rescaling of the zero-shot predictions?
#   sbatch slurm/55_rescale_control.sbatch
# One task per fold. Each does ONE forward pass over the target training period to fit the
# correction; the rescoring is closed-form arithmetic on the saved per-gauge tables, so
# there is no second pass and no training. Well inside 4 hours, which also keeps it on gpu4.
#
# It needs a GPU. Run on the login node it is a 20,052-batch forward pass on CPU that does
# not finish, and the login node here has no CUDA at all.
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --mem=96G --time=02:00:00
#SBATCH --constraint="v100|a100" --gres=gpu:1
#SBATCH --array=0-4
#SBATCH -J gmts_rescale -o logs/rescale_%A_%a.out -e logs/rescale_%A_%a.err
set -eo pipefail
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$REPO_ROOT/configs/phase1.yaml" ]] || { echo "not the repo: $REPO_ROOT" >&2; exit 2; }
cd "$REPO_ROOT"; mkdir -p logs
source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

FOLD="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG="${CONFIG:-configs/phase1_runB_v2.yaml}"
RUN="${RUN:-outputs/v2_runB}"
# Per-fold output directories, merged by the analysis afterwards. Five array tasks writing
# one CSV would race, and the loser's rows would vanish without an error.
OUT="${OUT_BASE:-outputs/v2_rescale_control}/fold$FOLD"

echo "fold=$FOLD config=$CONFIG run=$RUN out=$OUT node=$(hostname)"
"$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(), 'no GPU on this node'"

"$PYTHON_BIN" -u -m scripts.rescale_control \
  --config "$CONFIG" --folds "$FOLD" --run "$RUN" --out-dir "$OUT"
