#!/bin/bash
# Ensemble the five in-situ African models by averaging their daily predictions,
# matching how eval.africa reports (average predictions, then score once).
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --mem=96G --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH -J gmts_afens -o logs/afens_%j.out -e logs/afens_%j.err
set -eo pipefail
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$REPO_ROOT/configs/phase1.yaml" ]] || { echo "not the repo: $REPO_ROOT" >&2; exit 2; }
cd "$REPO_ROOT"; mkdir -p logs
source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
CONFIG="${CONFIG:-configs/phase1_runB.yaml}"
# Pass through the run location as well as the config. Without these the script's defaults
# win and it silently ensembles a different set of folds than the caller meant, writing over
# that set's summary. The learning-rate re-run needs a second, separately located set.
GLOB=(); [[ -n "${INSITU_GLOB:-}" ]] && GLOB=(--insitu-glob "$INSITU_GLOB")
PRE=();  [[ -n "${PRETRAIN_ROOT:-}" ]] && PRE=(--pretrain-root "$PRETRAIN_ROOT")
OUT=();  [[ -n "${OUT_DIR:-}" ]]     && OUT=(--out-dir "$OUT_DIR")
echo "config=$CONFIG glob=${INSITU_GLOB:-default} out=${OUT_DIR:-default} node=$(hostname)"
# Assert the GPU before starting. On a login node this is a many-thousand-batch forward pass
# on CPU that does not finish, and it slows the node for everyone rather than failing.
"$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(), 'no GPU on this node'"
"$PYTHON_BIN" -u -m scripts.africa_insitu_ensemble --config "$CONFIG" \
  "${GLOB[@]}" "${PRE[@]}" "${OUT[@]}"
