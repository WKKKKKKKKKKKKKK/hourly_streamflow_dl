#!/bin/bash
#SBATCH -J baseline_sweep
#SBATCH -o logs/sweep_agent_%A_%a.out
#SBATCH -e logs/sweep_agent_%A_%a.err
#SBATCH --constraint="a100|v100"
#SBATCH --time=10-00:00:00
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=1-8%8

# If your cluster does not recognize gpu:a100, try one of these patterns:
#   #SBATCH --gres=gpu:1
#   #SBATCH --constraint=a100
#   #SBATCH -p a100

set -eo pipefail

EXP_DIR="/home/kongw0a/MTS_LSTM/experiment_withcursor/BaselineLSTM"
cd "$EXP_DIR"

SWEEP_ID="${SWEEP_ID:-${1:-}}"
if [[ -z "${SWEEP_ID}" ]]; then
  echo "ERROR: Missing SWEEP_ID. Usage: sbatch --array=1-8%8 sweep_agent_array.sh <entity/project/sweep_id>"
  echo "Or: export SWEEP_ID=<...> and sbatch --array=1-8%8 sweep_agent_array.sh"
  exit 2
fi

# Allow passing a short sweep id (e.g. qap5q0kj). In that case we need entity+project.
export WANDB_PROJECT="${WANDB_PROJECT:-LSTM_Streamflow}"
if [[ "${SWEEP_ID}" != */*/* ]]; then
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    SWEEP_ID_BUILT="${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
    SWEEP_ID="${SWEEP_ID_BUILT}"
  else
    echo "ERROR: SWEEP_ID must be full form <entity>/<project>/<sweep_id> or set WANDB_ENTITY. Got: $SWEEP_ID"
    echo "Example: export WANDB_ENTITY=your_entity && sbatch --array=1-2 sweep_agent_array.sh $SWEEP_ID"
    exit 2
  fi
fi

export BASHRCSOURCED="${BASHRCSOURCED:-1}"
source ~/.bashrc
conda activate /ibex/user/kongw0a/conda-environments/hydroenv

# Ensure we run from the submission directory (SLURM copies scripts to /var/spool)
mkdir -p logs runs

# Date tag for output-dir templates
export RUN_DATE="$(date +%Y%m%d)"

# Optional: set WANDB_MODE=offline if compute nodes have no internet.
export WANDB_MODE="${WANDB_MODE:-online}"

echo "Starting W&B agent on host $(hostname), task ${SLURM_ARRAY_TASK_ID:-0}, sweep ${SWEEP_ID}"

# Set WANDB_AGENT_COUNT=1 for 1-combo-per-job behavior (array task ~= one hyperparameter combo).
# Leave as 0 to keep the agent running and pulling new configs until the sweep finishes.
WANDB_AGENT_COUNT="${WANDB_AGENT_COUNT:-0}"
AGENT_ARGS=()
if [[ "${WANDB_AGENT_COUNT}" != "0" ]]; then
  AGENT_ARGS+=(--count "${WANDB_AGENT_COUNT}")
fi

# Prevent the agent from shutting down after a few quick failures.
export WANDB_AGENT_DISABLE_FLAPPING="${WANDB_AGENT_DISABLE_FLAPPING:-true}"

wandb agent "${SWEEP_ID}" "${AGENT_ARGS[@]}"
