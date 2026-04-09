#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --constraint=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --array=1-12%3
#SBATCH -J mts100_tune
#SBATCH -o slurm-mts100_tune_%A_%a.out
#SBATCH -e slurm-mts100_tune_%A_%a.err

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/.." && pwd)"
CODE_DIR="$EXPERIMENT_DIR/code"

cd "$CODE_DIR"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "python is not available in PATH. Set PYTHON_BIN to your interpreter." >&2
  exit 2
fi

WANDB_BIN="${WANDB_BIN:-$(command -v wandb || true)}"
if [[ -n "${WANDB_BIN}" ]]; then
  "${WANDB_BIN}" online || true
fi

export WANDB_DIR="${WANDB_DIR:-$EXPERIMENT_DIR/wandb}"
mkdir -p "$WANDB_DIR"
BASE_OUT="${BASE_OUT:-$EXPERIMENT_DIR/training_runs/reproduced_100stations_tuning}"
mkdir -p "$BASE_OUT"

CONFIG="${CONFIG:-$SCRIPT_DIR/tuning_500stations_prevbest.tsv}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/selected_stn_data_100stations_west10_east90_proposal_boxes_part*.nc}"
STATIC_PATH="${STATIC_PATH:-$REPO_ROOT/data/static_h_topo_priority27.csv}"

if ! compgen -G "$DATA_PATH" > /dev/null; then
  echo "No dataset files matched DATA_PATH=$DATA_PATH" >&2
  exit 2
fi

if [[ ! -f "$STATIC_PATH" ]]; then
  echo "Static attribute file not found: $STATIC_PATH" >&2
  exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Tuning config file not found: $CONFIG" >&2
  exit 2
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is required" >&2
  exit 2
fi

line=$(awk -F $'\t' -v idx="$SLURM_ARRAY_TASK_ID" 'NR>1 && $1==idx {print $0}' "$CONFIG")
if [[ -z "$line" ]]; then
  echo "No config for idx=$SLURM_ARRAY_TASK_ID" >&2
  exit 3
fi

IFS=$'\t' read -r idx dropout hidden_size batch_size lookback_hourly lookback_daily lr_schedule epochs patience <<< "$line"
run_tag="idx${idx}_bs${batch_size}_do${dropout}_hs${hidden_size}_H${lookback_hourly}_D${lookback_daily}"
out_dir="$BASE_OUT/$run_tag"
mkdir -p "$out_dir"

"$PYTHON_BIN" -u Train.py \
  --data-path "$DATA_PATH" \
  --static-path "$STATIC_PATH" \
  --batch-size "$batch_size" \
  --num-workers 3 \
  --dropout "$dropout" \
  --hidden-size-daily "$hidden_size" \
  --hidden-size-hourly "$hidden_size" \
  --lookback-hourly "$lookback_hourly" \
  --lookback-daily "$lookback_daily" \
  --lr 5e-4 \
  --lr-schedule "$lr_schedule" \
  --reg-lambda 1.0 \
  --loss nse_loss \
  --num-layers 1 \
  --epochs "$epochs" \
  --early-stopping \
  --patience "$patience" \
  --seed 42 \
  --output-dir "$out_dir" \
  --no-resume \
  --wandb true \
  --wandb-project MTSLSTM_Streamflow \
  --wandb-run-name "mtslstm_100stations_tune_${run_tag}" \
  --wandb-mode online
